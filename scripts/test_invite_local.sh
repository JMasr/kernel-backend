#!/usr/bin/env bash
# =============================================================================
# test_invite_local.sh — Test the invitation flow without needing to check email
#
# Usage:
#   bash scripts/test_invite_local.sh [email_to_invite]
#
# What it does:
#   1. Reads ADMIN_EMAIL + ADMIN_PASS from .env
#   2. Logs in to POST /auth/login → gets JWT
#   3. Fetches the first available org from GET /organizations
#   4. Creates an invitation via POST /admin/invitations
#   5. Prints the invite URL to open directly in the browser
#
# Requirements: curl, jq
# =============================================================================
set -euo pipefail

API="http://localhost:8000"
ENV_FILE="$(dirname "$0")/../.env"

# ── Read credentials from .env ────────────────────────────────────────────────
if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: .env not found at $ENV_FILE"
  exit 1
fi

ADMIN_EMAIL=$(grep -E '^ADMIN_EMAIL=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
ADMIN_PASS=$(grep -E '^ADMIN_PASS=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
FRONTEND_URL=$(grep -E '^FRONTEND_BASE_URL=' "$ENV_FILE" | cut -d= -f2- | tr -d '"' | tr -d "'")
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"

if [[ -z "$ADMIN_EMAIL" || -z "$ADMIN_PASS" ]]; then
  echo "ERROR: ADMIN_EMAIL or ADMIN_PASS not set in .env"
  exit 1
fi

# ── Invite target: use arg or default to a test address ───────────────────────
INVITE_EMAIL="${1:-test-invite@example.com}"

echo ""
echo "▶ Kernel Security — Local Invitation Test"
echo "  API:          $API"
echo "  Frontend:     $FRONTEND_URL"
echo "  Admin:        $ADMIN_EMAIL"
echo "  Inviting:     $INVITE_EMAIL"
echo ""

# ── Step 1: Log in ────────────────────────────────────────────────────────────
echo "1/4  Logging in..."
LOGIN=$(curl -sf -X POST "$API/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$ADMIN_EMAIL\", \"password\": \"$ADMIN_PASS\"}")

TOKEN=$(echo "$LOGIN" | jq -r '.access_token')
if [[ -z "$TOKEN" || "$TOKEN" == "null" ]]; then
  echo "ERROR: Login failed. Response: $LOGIN"
  exit 1
fi
echo "     ✓ JWT obtained"

AUTH="Authorization: Bearer $TOKEN"

# ── Step 2: Get first org ─────────────────────────────────────────────────────
echo "2/4  Fetching organizations..."
ORGS=$(curl -sf -X GET "$API/organizations" -H "$AUTH")
ORG_ID=$(echo "$ORGS" | jq -r '.items[0].org_id // .items[0].id // empty')
ORG_NAME=$(echo "$ORGS" | jq -r '.items[0].name // "Unknown"')

if [[ -z "$ORG_ID" ]]; then
  echo "ERROR: No organizations found. Create one first via the admin panel."
  echo "       GET /organizations returned: $ORGS"
  exit 1
fi
echo "     ✓ Using org: \"$ORG_NAME\" ($ORG_ID)"

# ── Step 3: Create invitation ─────────────────────────────────────────────────
echo "3/4  Creating invitation for $INVITE_EMAIL..."
INVITE=$(curl -sf -X POST "$API/admin/invitations" \
  -H "$AUTH" \
  -H "Content-Type: application/json" \
  -d "{\"email\": \"$INVITE_EMAIL\", \"org_id\": \"$ORG_ID\"}")

INVITE_TOKEN=$(echo "$INVITE" | jq -r '.token // empty')
INVITE_STATUS=$(echo "$INVITE" | jq -r '.status // empty')

if [[ -z "$INVITE_TOKEN" ]]; then
  echo "ERROR: Failed to create invitation. Response: $INVITE"
  exit 1
fi
echo "     ✓ Invitation created (status: $INVITE_STATUS)"

# ── Step 4: Print the URL ─────────────────────────────────────────────────────
echo "4/4  Done!"
echo ""
echo "  ┌─────────────────────────────────────────────────────────────────"
echo "  │  Open this URL in the browser to test the invite page:"
echo "  │"
echo "  │  $FRONTEND_URL/invite/$INVITE_TOKEN"
echo "  │"
echo "  │  (Next.js will redirect to /es/invite/$INVITE_TOKEN or /en/...)"
echo "  └─────────────────────────────────────────────────────────────────"
echo ""
echo "  Token raw value: $INVITE_TOKEN"
echo "  Org:             $ORG_NAME ($ORG_ID)"
echo "  Invited email:   $INVITE_EMAIL"
echo ""
