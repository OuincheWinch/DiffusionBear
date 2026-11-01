import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("DiffusionBear React Runtime Crash:", error, errorInfo);
    this.setState({ errorInfo });
  }

  handleReload = () => {
    window.location.reload();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{
          minHeight: "100vh",
          backgroundColor: "#0f1117",
          color: "#e2e8f0",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          padding: "2rem",
          fontFamily: "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
          boxSizing: "border-box",
        }}>
          <div style={{
            maxWidth: "640px",
            width: "100%",
            backgroundColor: "#1a1d26",
            border: "1px solid #dc2626",
            borderRadius: "12px",
            padding: "2rem",
            boxShadow: "0 20px 25px -5px rgba(0, 0, 0, 0.5)",
          }}>
            <h2 style={{ color: "#ef4444", marginTop: 0, display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              DiffusionBear Interface Recovery
            </h2>
            <p style={{ color: "#94a3b8", lineHeight: 1.6 }}>
              A client-side interface error occurred. Rather than leaving you with a black screen, this recovery boundary caught the exception.
            </p>
            <div style={{
              backgroundColor: "#0d0f14",
              padding: "1rem",
              borderRadius: "8px",
              border: "1px solid #2d3748",
              fontFamily: "monospace",
              fontSize: "0.85rem",
              color: "#f87171",
              overflowX: "auto",
              whiteSpace: "pre-wrap",
              margin: "1rem 0",
            }}>
              {this.state.error?.toString()}
            </div>
            <button
              type="button"
              onClick={this.handleReload}
              style={{
                backgroundColor: "#2563eb",
                color: "#ffffff",
                border: "none",
                padding: "0.75rem 1.5rem",
                borderRadius: "8px",
                fontWeight: 600,
                cursor: "pointer",
                fontSize: "0.95rem",
                transition: "background-color 0.15s ease",
                display: "inline-flex",
                alignItems: "center",
                gap: "0.5rem",
              }}
              onMouseOver={(e) => (e.currentTarget.style.backgroundColor = "#1d4ed8")}
              onMouseOut={(e) => (e.currentTarget.style.backgroundColor = "#2563eb")}
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>
              Refresh Studio
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
