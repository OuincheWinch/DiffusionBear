import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './App.css'
import App from './App.jsx'
import ErrorBoundary from './components/ErrorBoundary.jsx'
import { I18nProvider } from './i18n/I18nContext.jsx'
import { ThemeProvider } from './theme.js'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ErrorBoundary>
      <I18nProvider>
        <ThemeProvider>
          <App />
        </ThemeProvider>
      </I18nProvider>
    </ErrorBoundary>
  </StrictMode>,
)

