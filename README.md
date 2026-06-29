# Azure OpenAI PTU autoscaler

Small scheduled job that opportunistically scales an existing Azure OpenAI / Microsoft Foundry provisioned deployment toward a target PTU count when regional physical capacity becomes available.

This is useful when quota is approved but a resize fails because the selected region is temporarily capacity constrained. The job is idempotent and safe to schedule: it reads the current deployment size, checks model capacity, attempts a resize only when there is headroom, and uses jittered exponential backoff for transient ARM throttling or service errors.

## What it does

1. Reads the current deployment PTU capacity.
2. Exits if the deployment is already at or above `TARGET_PTU`.
3. Checks `Microsoft.CognitiveServices/modelCapacities` for the requested model, region, and SKU.
4. If `GRAB_INCREMENTALLY=true`, scales only by the capacity currently reported as available so partial PTUs are captured quickly.
5. Uses `Retry-After`, `retry-after-ms`, and jittered exponential backoff for 408/409/429/5xx responses.
6. Uses a local lock file to avoid overlapping scheduled runs.

Capacity is still not guaranteed. This automates the polling and resize attempt.

## Prerequisites

- Python 3.10+
- An identity that can read and update the Azure OpenAI resource, for example **Cognitive Services Contributor** on the resource or resource group
- One authentication method supported by `DefaultAzureCredential`, such as `az login`, managed identity, or service principal environment variables

```bash
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` for local use, or set the same values as scheduler environment variables. Do not commit `.env`.

Required values:

| Variable | Description |
| --- | --- |
| `SUBSCRIPTION_ID` | Azure subscription ID |
| `RESOURCE_GROUP` | Resource group containing the Azure OpenAI / Foundry resource |
| `ACCOUNT_NAME` | Azure OpenAI / Foundry resource name |
| `DEPLOYMENT_NAME` | Existing deployment to resize |
| `MODEL_NAME` / `MODEL_VERSION` | Model deployed, for example `gpt-4o` / `2024-11-20` |
| `REGION` | Azure region name without spaces, for example `germanywestcentral` |
| `SKU_NAME` | `ProvisionedManaged`, `DataZoneProvisionedManaged`, or `GlobalProvisionedManaged` |
| `TARGET_PTU` | Desired final PTU count |

## Run once

```bash
python ptu_autoscale.py
```

Exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Target reached or resize succeeded |
| `2` | Not at target yet, but safe for scheduler retry |
| `1` | Fatal configuration/auth/non-retryable error |

## Schedule

Use a 15-30 minute cadence. Avoid one-minute schedules; capacity does not appear that quickly and frequent polling increases the chance of ARM throttling.

Windows Task Scheduler:

```powershell
schtasks /create /sc minute /mo 15 /tn "PTU-Autoscale" /tr "python C:\path\ptu_autoscale.py"
```

Linux cron:

```cron
*/15 * * * * /usr/bin/python3 /path/ptu_autoscale.py >> /var/log/ptu-autoscale.log 2>&1
```

## Backoff policy

The script retries transient ARM failures only. Capacity-unavailable failures are not retried in a tight loop; the job exits with code `2` and lets the scheduler retry later.

Defaults:

| Variable | Default |
| --- | --- |
| `MAX_ATTEMPTS` | `5` |
| `BACKOFF_INITIAL_SECONDS` | `30` |
| `BACKOFF_MAX_SECONDS` | `900` |
| `BACKOFF_MULTIPLIER` | `2` |
| `BACKOFF_JITTER_RATIO` | `0.2` |
| `POLL_INTERVAL_SECONDS` | `30` |

If Azure returns `Retry-After` or `retry-after-ms`, that value is honored before calculating exponential backoff.
