# Your own LLM on Oracle Cloud (Always Free)

Runs an open-source model (default `qwen2.5:7b-instruct`) with Ollama on a free Ampere A1 VM. Your data stays on your server: no third-party AI provider sees column names or samples. Groq stays configured as a backup for when the server is slow or down.

```
Render (AskData API) --HTTPS + Bearer token--> Caddy on the VM --> Ollama (127.0.0.1 only)
        \--> Groq, if the VM doesn't answer within 60 s
```

## 1. Create the VM

Oracle Cloud console: **Compute > Instances > Create instance**

| Setting | Value |
|---|---|
| Image | **Canonical Ubuntu 24.04** (or 22.04) |
| Shape | **Ampere > VM.Standard.A1.Flex**, **4 OCPUs, 24 GB memory** (the Always Free maximum) |
| Networking | Keep "Assign a public IPv4 address" on |
| SSH keys | **Generate a key pair for me** and **download the private key** |
| Boot volume | Default (about 47 GB) is enough |

If you see **"Out of capacity for shape VM.Standard.A1.Flex"**, pick a different Availability Domain, or try again later. Free Ampere capacity comes and goes.

## 2. Open ports 80 and 443 in Oracle's network

Instance page > **Subnet** link > **Security Lists** > Default Security List > **Add Ingress Rules**:

| Source CIDR | IP protocol | Destination port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

Port 80 is only used for the HTTPS certificate. Every request still needs the token.

## 3. Run the setup script

From your PC (PowerShell), in the folder with the downloaded key:

```powershell
ssh -i .\ssh-key-XXXX.key ubuntu@<PUBLIC-IP>
```

On the VM:

```bash
curl -fsSLO https://raw.githubusercontent.com/Ubdale/smartbot/main/deploy/oracle/setup.sh
sudo bash setup.sh
```

It takes about 5-10 minutes (most of it downloading the model, ~4.7 GB). At the end it prints the values for Render.

## 4. Point Render at it

Render > your service > **Environment**: set `LLM_PROVIDER=ollama`, `LLM_MODEL`, `OLLAMA_BASE_URL`, `OLLAMA_API_KEY` and `LLM_FALLBACK_PROVIDER=groq` as printed, then save. Keep `GROQ_API_KEY` so the backup works. `GEMINI_API_KEY` is no longer needed.

## Speed and choosing a model

A 4-core ARM CPU runs a 7B model at roughly 15-40 s per question; the repeated part of the prompt is cached, so later questions are faster. If answers often take longer than 60 s, AskData quietly uses Groq for that question. To trade some accuracy for speed:

```bash
sudo MODEL=qwen2.5:3b-instruct bash setup.sh   # then set LLM_MODEL=qwen2.5:3b-instruct on Render
```

## Maintenance

- Status: `systemctl status ollama caddy`
- Logs: `sudo journalctl -u ollama -u caddy --no-pager | tail -50`
- Token: `sudo cat /etc/askdata-llm-token`. To change it, delete that file and rerun the script, then update Render.
- Oracle reclaims Always Free VMs that stay almost completely idle for 7 days. Regular use keeps it alive.
