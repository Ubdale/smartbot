#!/usr/bin/env bash
# Run AskData's own LLM on an Oracle Cloud "Always Free" Ampere A1 VM (Ubuntu, ARM).
#
#   sudo bash setup.sh                      # default model qwen2.5:7b-instruct
#   sudo MODEL=qwen2.5:3b-instruct bash setup.sh
#
# What it does:
#   1. Installs Ollama, listening on 127.0.0.1 only (Ollama has no authentication).
#   2. Puts Caddy in front: free HTTPS (Let's Encrypt via <ip>.sslip.io) and a check that
#      every request carries "Authorization: Bearer <token>". Everything else gets 401.
#   3. Opens ports 80/443 in the VM's own firewall (Oracle images block them by default).
# At the end it prints OLLAMA_BASE_URL and OLLAMA_API_KEY for Render.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then echo "Run with sudo: sudo bash $0" >&2; exit 1; fi

MODEL="${MODEL:-qwen2.5:7b-instruct}"
PUBLIC_IP="${PUBLIC_IP:-$(curl -fsS https://api.ipify.org)}"
DOMAIN="${DOMAIN:-${PUBLIC_IP//./-}.sslip.io}"
TOKEN_FILE=/etc/askdata-llm-token
if [[ -s $TOKEN_FILE ]]; then TOKEN=$(cat "$TOKEN_FILE"); else
  TOKEN=$(openssl rand -hex 32); install -m 600 /dev/null "$TOKEN_FILE"; echo "$TOKEN" > "$TOKEN_FILE"
fi

echo "==> 1/4 Ollama"
if ! command -v ollama >/dev/null; then curl -fsSL https://ollama.com/install.sh | sh; fi
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/askdata.conf <<EOF
[Service]
# Reachable only from this machine; Caddy is the only way in.
Environment="OLLAMA_HOST=127.0.0.1:11434"
# Keep the model in memory (24 GB RAM is plenty): no 20 s reload after idle periods.
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_NUM_PARALLEL=1"
EOF
systemctl daemon-reload
systemctl enable --now ollama
systemctl restart ollama
sleep 3
ollama pull "$MODEL"

echo "==> 2/4 Firewall (ports 80, 443)"
# Oracle images end INPUT with a catch-all REJECT; the ACCEPTs must come before it.
# Its position differs between images, so look it up instead of assuming a line number.
for port in 443 80; do
  while iptables -D INPUT -p tcp --dport "$port" -m state --state NEW -j ACCEPT 2>/dev/null; do :; done
  reject_at=$(iptables -L INPUT -n --line-numbers | awk '$2 == "REJECT" { print $1; exit }')
  iptables -I INPUT "${reject_at:-1}" -p tcp --dport "$port" -m state --state NEW -j ACCEPT
done
if command -v netfilter-persistent >/dev/null; then netfilter-persistent save; fi

echo "==> 3/4 Caddy (HTTPS + token check)"
if ! command -v caddy >/dev/null; then
  apt-get update -y
  apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl gnupg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key \
    | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt \
    > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y
  apt-get install -y caddy
fi
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
	@authorized header Authorization "Bearer $TOKEN"
	handle @authorized {
		reverse_proxy 127.0.0.1:11434 {
			header_up Host 127.0.0.1:11434
			transport http {
				read_timeout 300s
			}
		}
	}
	respond "unauthorized" 401
}
EOF
chown root:caddy /etc/caddy/Caddyfile
chmod 640 /etc/caddy/Caddyfile
systemctl enable --now caddy
systemctl reload caddy

echo "==> 4/4 Check"
sleep 5
code=$(curl -s -o /dev/null -w '%{http_code}' "https://$DOMAIN/api/tags" || true)
authed=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $TOKEN" "https://$DOMAIN/api/tags" || true)
echo "  without token: HTTP $code (expect 401)    with token: HTTP $authed (expect 200)"
if [[ $authed != 200 ]]; then
  echo "  HTTPS isn't up yet. Check the Oracle Security List allows TCP 80 and 443 (see README)," >&2
  echo "  then: sudo journalctl -u caddy --no-pager | tail -30" >&2
fi

cat <<EOF

Done. Add these to Render (Environment), then save:

  LLM_PROVIDER=ollama
  LLM_MODEL=$MODEL
  OLLAMA_BASE_URL=https://$DOMAIN
  OLLAMA_API_KEY=$TOKEN
  LLM_FALLBACK_PROVIDER=groq

Keep OLLAMA_API_KEY secret: it's the only thing protecting your model server.
EOF
