#!/usr/bin/env python3
"""
Opportunistically scale an Azure OpenAI / Microsoft Foundry provisioned deployment.

The script is designed to be scheduled. It avoids tight retry loops by using
jittered exponential backoff for transient ARM errors and by exiting cleanly
when physical PTU capacity is not currently available.
"""

from __future__ import annotations

import email.utils
import logging
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from azure.identity import DefaultAzureCredential
from requests import Response

ARM_SCOPE = "https://management.azure.com/.default"
ARM_ROOT = "https://management.azure.com"
API_VERSION = "2024-10-01"
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
CAPACITY_ERROR_MARKERS = (
    "capacity",
    "insufficient",
    "not enough",
    "unavailable",
    "invalid capacity",
    "quota",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("ptu-autoscale")


@dataclass(frozen=True)
class Config:
    subscription_id: str
    resource_group: str
    account_name: str
    deployment_name: str
    model_format: str
    model_name: str
    model_version: str
    region: str
    sku_name: str
    target_ptu: int
    min_step: int
    grab_incrementally: bool
    max_attempts: int
    backoff_initial_seconds: float
    backoff_max_seconds: float
    backoff_multiplier: float
    backoff_jitter_ratio: float
    poll_interval_seconds: float
    max_poll_seconds: float
    request_timeout_seconds: float
    lock_file: Path
    notify_webhook: str


class ArmError(Exception):
    def __init__(self, response: Response):
        self.response = response
        self.status_code = response.status_code
        self.payload = _safe_json(response)
        super().__init__(self.message)

    @property
    def message(self) -> str:
        if isinstance(self.payload, dict):
            error = self.payload.get("error", {})
            if isinstance(error, dict):
                code = error.get("code", "ARMError")
                msg = error.get("message", self.response.text)
                return f"{code}: {msg}"
        return self.response.text or f"HTTP {self.status_code}"


class TransientArmError(ArmError):
    pass


def env_required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def load_config() -> Config:
    account = env_required("ACCOUNT_NAME")
    deployment = env_required("DEPLOYMENT_NAME")
    lock_file = Path(
        os.getenv(
            "LOCK_FILE",
            str(Path(tempfile.gettempdir()) / f"ptu-autoscale-{account}-{deployment}.lock"),
        )
    )
    return Config(
        subscription_id=env_required("SUBSCRIPTION_ID"),
        resource_group=env_required("RESOURCE_GROUP"),
        account_name=account,
        deployment_name=deployment,
        model_format=os.getenv("MODEL_FORMAT", "OpenAI"),
        model_name=os.getenv("MODEL_NAME", "gpt-4o"),
        model_version=os.getenv("MODEL_VERSION", "2024-11-20"),
        region=os.getenv("REGION", "germanywestcentral").replace(" ", "").lower(),
        sku_name=os.getenv("SKU_NAME", "ProvisionedManaged"),
        target_ptu=env_int("TARGET_PTU", 150),
        min_step=env_int("MIN_STEP", 10),
        grab_incrementally=env_bool("GRAB_INCREMENTALLY", True),
        max_attempts=env_int("MAX_ATTEMPTS", 5),
        backoff_initial_seconds=env_float("BACKOFF_INITIAL_SECONDS", 30),
        backoff_max_seconds=env_float("BACKOFF_MAX_SECONDS", 900),
        backoff_multiplier=env_float("BACKOFF_MULTIPLIER", 2),
        backoff_jitter_ratio=env_float("BACKOFF_JITTER_RATIO", 0.2),
        poll_interval_seconds=env_float("POLL_INTERVAL_SECONDS", 30),
        max_poll_seconds=env_float("MAX_POLL_SECONDS", 1800),
        request_timeout_seconds=env_float("REQUEST_TIMEOUT_SECONDS", 60),
        lock_file=lock_file,
        notify_webhook=os.getenv("NOTIFY_WEBHOOK", ""),
    )


def _safe_json(response: Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _retry_after_seconds(headers: requests.structures.CaseInsensitiveDict[str]) -> float | None:
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return max(float(retry_after_ms) / 1000, 0)
        except ValueError:
            pass

    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None
    try:
        return max(float(retry_after), 0)
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(retry_after)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max((parsed - datetime.now(timezone.utc)).total_seconds(), 0)


def _backoff_delay(config: Config, attempt: int, response: Response | None = None) -> float:
    if response is not None:
        retry_after = _retry_after_seconds(response.headers)
        if retry_after is not None:
            return min(retry_after, config.backoff_max_seconds)

    raw_delay = config.backoff_initial_seconds * (config.backoff_multiplier ** max(attempt - 1, 0))
    capped = min(raw_delay, config.backoff_max_seconds)
    jitter = capped * config.backoff_jitter_ratio
    return max(random.uniform(capped - jitter, capped + jitter), 0)


def _is_capacity_error(error: ArmError) -> bool:
    text = error.message.lower()
    return any(marker in text for marker in CAPACITY_ERROR_MARKERS)


def _token(credential: DefaultAzureCredential) -> str:
    return credential.get_token(ARM_SCOPE).token


def arm_request(
    config: Config,
    credential: DefaultAzureCredential,
    method: str,
    url: str,
    *,
    expected: set[int],
    payload: dict[str, Any] | None = None,
    retry_transient: bool = True,
) -> Response:
    for attempt in range(1, config.max_attempts + 1):
        response = requests.request(
            method,
            url,
            headers={
                "Authorization": f"Bearer {_token(credential)}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.request_timeout_seconds,
        )
        if response.status_code in expected:
            return response

        error_cls = TransientArmError if response.status_code in TRANSIENT_STATUS_CODES else ArmError
        error = error_cls(response)
        if not retry_transient or not isinstance(error, TransientArmError) or attempt == config.max_attempts:
            raise error

        delay = _backoff_delay(config, attempt, response)
        log.warning(
            "ARM %s %s returned %s (%s). Retry %s/%s in %.1fs.",
            method,
            url,
            response.status_code,
            error.message,
            attempt + 1,
            config.max_attempts,
            delay,
        )
        time.sleep(delay)

    raise RuntimeError("unreachable")


def deployment_url(config: Config) -> str:
    return (
        f"{ARM_ROOT}/subscriptions/{config.subscription_id}"
        f"/resourceGroups/{config.resource_group}"
        f"/providers/Microsoft.CognitiveServices/accounts/{config.account_name}"
        f"/deployments/{config.deployment_name}"
        f"?api-version={API_VERSION}"
    )


def model_capacities_url(config: Config) -> str:
    query = urlencode(
        {
            "api-version": API_VERSION,
            "modelFormat": config.model_format,
            "modelName": config.model_name,
            "modelVersion": config.model_version,
        }
    )
    return f"{ARM_ROOT}/subscriptions/{config.subscription_id}/providers/Microsoft.CognitiveServices/modelCapacities?{query}"


def get_deployment(config: Config, credential: DefaultAzureCredential) -> dict[str, Any]:
    response = arm_request(config, credential, "GET", deployment_url(config), expected={200})
    return response.json()


def get_available_capacity(config: Config, credential: DefaultAzureCredential) -> int | None:
    try:
        response = arm_request(config, credential, "GET", model_capacities_url(config), expected={200})
    except ArmError as error:
        log.warning("Capacity pre-check failed (%s). Will try blind resize.", error.message)
        return None

    best = 0
    for item in response.json().get("value", []):
        location = str(item.get("location", "")).replace(" ", "").lower()
        props = item.get("properties", {})
        if location != config.region:
            continue
        if props.get("skuName") != config.sku_name:
            continue
        best = max(best, int(props.get("availableCapacity", 0) or 0))

    log.info(
        "Region %s reports ~%s capacity units for %s %s (%s).",
        config.region,
        best,
        config.model_name,
        config.model_version,
        config.sku_name,
    )
    return best


def build_resize_body(existing: dict[str, Any], config: Config, new_capacity: int) -> dict[str, Any]:
    existing_props = existing.get("properties", {})
    properties: dict[str, Any] = {
        "model": existing_props.get(
            "model",
            {
                "format": config.model_format,
                "name": config.model_name,
                "version": config.model_version,
            },
        )
    }
    for optional_field in ("versionUpgradeOption", "raiPolicyName"):
        value = existing_props.get(optional_field)
        if value:
            properties[optional_field] = value

    return {"sku": {"name": config.sku_name, "capacity": new_capacity}, "properties": properties}


def poll_operation(
    config: Config,
    credential: DefaultAzureCredential,
    initial_response: Response,
) -> None:
    poll_url = (
        initial_response.headers.get("Azure-AsyncOperation")
        or initial_response.headers.get("Operation-Location")
        or initial_response.headers.get("Location")
    )
    if not poll_url:
        return

    deadline = time.monotonic() + config.max_poll_seconds
    while time.monotonic() < deadline:
        delay = _retry_after_seconds(initial_response.headers)
        if delay is None:
            delay = random.uniform(
                config.poll_interval_seconds * 0.8,
                config.poll_interval_seconds * 1.2,
            )
        time.sleep(min(delay, config.backoff_max_seconds))

        response = arm_request(config, credential, "GET", poll_url, expected={200, 201, 202})
        payload = _safe_json(response) or {}
        status = str(payload.get("status") or payload.get("properties", {}).get("provisioningState") or "").lower()
        if status in {"succeeded", "success"}:
            return
        if status in {"failed", "canceled", "cancelled"}:
            raise ArmError(response)

        log.info("Resize still running (operation status=%s).", status or "unknown")

    raise TimeoutError(f"Resize operation did not finish within {config.max_poll_seconds}s")


def resize_deployment(
    config: Config,
    credential: DefaultAzureCredential,
    existing: dict[str, Any],
    new_capacity: int,
) -> dict[str, Any]:
    response = arm_request(
        config,
        credential,
        "PUT",
        deployment_url(config),
        expected={200, 201, 202},
        payload=build_resize_body(existing, config, new_capacity),
        retry_transient=True,
    )
    poll_operation(config, credential, response)
    return get_deployment(config, credential)


def notify(config: Config, message: str) -> None:
    if not config.notify_webhook:
        return
    try:
        requests.post(config.notify_webhook, json={"text": message}, timeout=10)
    except requests.RequestException as error:
        log.warning("Notification failed: %s", error)


def acquire_lock(config: Config) -> int | None:
    try:
        fd = os.open(config.lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log.info("Another run appears active (%s). Exiting without action.", config.lock_file)
        return None

    os.write(fd, str(os.getpid()).encode("utf-8"))
    return fd


def release_lock(config: Config, fd: int) -> None:
    os.close(fd)
    try:
        config.lock_file.unlink()
    except FileNotFoundError:
        pass


def main() -> int:
    config = load_config()
    lock_fd = acquire_lock(config)
    if lock_fd is None:
        return 2

    try:
        credential = DefaultAzureCredential()
        existing = get_deployment(config, credential)
        current_capacity = int(existing.get("sku", {}).get("capacity", 0) or 0)
        log.info(
            "Deployment %s current capacity=%s target=%s.",
            config.deployment_name,
            current_capacity,
            config.target_ptu,
        )

        if current_capacity >= config.target_ptu:
            log.info("Already at or above target. Nothing to do.")
            return 0

        available_capacity = get_available_capacity(config, credential)
        if config.grab_incrementally and available_capacity is not None:
            if available_capacity < config.min_step:
                log.info(
                    "Only %s capacity units available (< MIN_STEP=%s). Scheduler can retry later.",
                    available_capacity,
                    config.min_step,
                )
                return 2
            new_capacity = min(config.target_ptu, current_capacity + available_capacity)
        else:
            new_capacity = config.target_ptu

        if new_capacity <= current_capacity:
            log.info("No capacity increase to attempt.")
            return 2

        log.info("Attempting resize from %s to %s capacity units.", current_capacity, new_capacity)
        try:
            updated = resize_deployment(config, credential, existing, new_capacity)
        except ArmError as error:
            if _is_capacity_error(error):
                log.warning("Capacity not available yet: %s. Scheduler can retry later.", error.message)
                return 2
            raise

        final_capacity = int(updated.get("sku", {}).get("capacity", new_capacity) or new_capacity)
        message = (
            f"{config.deployment_name} scaled to {final_capacity} capacity units "
            f"in {config.region} ({final_capacity}/{config.target_ptu})."
        )
        log.info(message)
        notify(config, message)
        return 0 if final_capacity >= config.target_ptu else 2
    finally:
        release_lock(config, lock_fd)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ArmError, TimeoutError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)
    except requests.RequestException as exc:
        log.error("HTTP client error: %s", exc)
        sys.exit(1)
