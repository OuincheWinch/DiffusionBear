import { useState } from "react";
import { useI18n } from "../../i18n/I18nContext";

const TABS = [
  { id: "browsers", labelKey: "settings.api.guide.tabBrowsers", vars: { browsers: "Chrome, Firefox, Opera, Safari" } },
  { id: "agents", labelKey: "settings.api.guide.tabAgents", vars: { curl: "curl" } },
  { id: "blocks", labelKey: "settings.api.guide.tabBlocks" },
];

export default function ApiConnectionGuide() {
  const { t } = useI18n();
  const [activeTab, setActiveTab] = useState("browsers");

  return (
    <details className="api-guide-card">
      <summary className="api-guide-card__summary">
        <span className="api-guide-card__title">
          {t("settings.api.guide.title", { browsers: "Chrome, Firefox, Opera, Safari" })}
        </span>
        <span className="api-guide-card__subtitle">{t("settings.api.guide.subtitle")}</span>
      </summary>

      <div className="api-guide-card__body">
        <div
          className="api-guide-tabs"
          role="tablist"
          aria-label={t("settings.api.guide.title", { browsers: "Chrome, Firefox, Opera, Safari" })}
        >
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={activeTab === entry.id}
              className={`api-guide-tab${activeTab === entry.id ? " is-active" : ""}`}
              onClick={() => setActiveTab(entry.id)}
            >
              {t(entry.labelKey, entry.vars)}
            </button>
          ))}
        </div>

        {activeTab === "browsers" && (
          <div className="api-guide-panel" role="tabpanel">
            <p className="params-hint">{t("settings.api.guide.browsersIntro")}</p>

            <div className="api-guide-steps">
              <div className="api-guide-step">
                <strong>1. {t("settings.api.guide.stepHostTitle")}</strong>
                <p>
                  {t("settings.api.guide.stepHostBody", {
                    sampleLabel: '"Safari iPad"',
                    tailscale: "Tailscale Serve",
                    app: "DiffusionBear.app",
                    loopback: "127.0.0.1",
                    flag: "--host 0.0.0.0",
                  })}
                </p>
                <pre className="api-guide-code"><code>{`# Recommended with DiffusionBear.app (HTTPS via Tailscale)\ntailscale serve --bg --https=443 http://127.0.0.1:8001\ntailscale serve status\n\n# Or direct LAN bind via CLI (plain HTTP)\ncd backend && DIFFUSIONBEAR_API_KEY="dbk_..." ../venv/bin/python main.py --host 0.0.0.0 --port 8001`}</code></pre>
              </div>
              <div className="api-guide-step">
                <strong>2. {t("settings.api.guide.stepConnectTitle", { browsers: "Chrome / Firefox / Opera / Safari" })}</strong>
                <p>
                  {t("settings.api.guide.stepConnectBody", {
                    httpsUrl: "https://<mac>.ts.net",
                    lanUrl: "http://<LAN-IP>:8001",
                  })}
                </p>
              </div>
            </div>

            <div className="api-browser-grid">
              <div className="api-browser-card">
                <strong>Google Chrome</strong>
                <p>
                  {t("settings.api.guide.browserChrome", {
                    httpsPrefix: "https://",
                    tailscale: "Tailscale Serve",
                    mixedContent: "Mixed Content",
                    lanUrl: "http://192.168.x.x:8001",
                    localStorage: "localStorage",
                  })}
                </p>
              </div>
              <div className="api-browser-card">
                <strong>Mozilla Firefox</strong>
                <p>
                  {t("settings.api.guide.browserFirefox", {
                    httpsUrl: "https://<mac>.ts.net",
                    ips: "192.168.x.x / 100.x.y.z",
                    doh: "DNS over HTTPS",
                    dotLocal: ".local",
                  })}
                </p>
              </div>
              <div className="api-browser-card">
                <strong>Opera</strong>
                <p>
                  {t("settings.api.guide.browserOpera", {
                    vpn: "Opera VPN",
                    tailscale: "Tailscale",
                    chrome: "Chrome",
                    mixedContent: "HTTPS Mixed Content",
                  })}
                </p>
              </div>
              <div className="api-browser-card">
                <strong>Apple Safari</strong>
                <p>
                  {t("settings.api.guide.browserSafari", {
                    tailscale: "Tailscale",
                    httpsUrl: "https://<mac>.ts.net",
                    localStorage: "localStorage",
                  })}
                </p>
              </div>
            </div>
          </div>
        )}

        {activeTab === "agents" && (
          <div className="api-guide-panel" role="tabpanel">
            <p className="params-hint">
              {t("settings.api.guide.agentsIntro", {
                apiWildcard: "/api/*",
                versionRoute: "GET /api/version",
                header1: "Authorization: Bearer",
                header2: "X-API-Key",
                query: "?token=",
                imgTag: "<img>",
              })}
            </p>
            <pre className="api-guide-code"><code>{`# Local or remote base URL (http://127.0.0.1:8001 or https://mac.tailnet.ts.net)\nexport DB_URL="http://127.0.0.1:8001"\nexport DB_KEY="dbk_your_managed_or_env_key"\n\n# 1. Queue a generation\ncurl -sS -X POST "$DB_URL/api/generate" \\\n  -H "Authorization: Bearer $DB_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"flux2-klein-4b","prompt":"a wooden cabin at dusk","width":512,"height":768,"steps":4}'\n\n# 2. Poll job status\ncurl -sS -H "X-API-Key: $DB_KEY" "$DB_URL/api/jobs/<job_id>"\n\n# 3. Download generated PNG bytes\ncurl -sS -H "Authorization: Bearer $DB_KEY" "$DB_URL/api/images/<image_id>/file" -o output.png`}</code></pre>
            <p className="params-hint">
              {t("settings.api.guide.agentsNote", {
                genRoute: "POST /api/generate",
                retryAfter: "Retry-After",
              })}
            </p>
          </div>
        )}

        {activeTab === "blocks" && (
          <div className="api-guide-panel" role="tabpanel">
            <ul className="api-guide-list">
              <li>
                <strong>{t("settings.api.mode")}:</strong>{" "}
                {t("settings.api.guide.blockMode", {
                  off: "OFF",
                  on: "ON",
                  loopback: "127.0.0.1",
                  apiWildcard: "/api/*",
                })}
              </li>
              <li>
                <strong>{t("settings.api.portTitle")}:</strong>{" "}
                {t("settings.api.guide.blockPort", { on: "ON" })}
              </li>
              <li>
                <strong>{t("settings.api.connectionTitle")}:</strong>{" "}
                {t("settings.api.guide.blockConnection", {
                  localStorage: "localStorage",
                  client: "CLIENT",
                })}
              </li>
              <li>
                <strong>{t("settings.api.managed")}:</strong>{" "}
                {t("settings.api.guide.blockKeys", {
                  envVar: "DIFFUSIONBEAR_API_KEY",
                  prefix: "dbk_...",
                  sha256: "SHA-256",
                })}
              </li>
              <li>
                <strong>{t("settings.api.guide.recoveryTitle")}:</strong>{" "}
                {t("settings.api.guide.blockRecovery", {
                  localStorage: "localStorage",
                  prefix: "dbk_...",
                  securityJson: "~/Library/Application Support/DiffusionBear/data/security.json",
                  enabledFalse: '"enabled": false',
                })}
              </li>
            </ul>
          </div>
        )}
      </div>
    </details>
  );
}
