import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Gemini Forex Autotrader',
  description:
    'Autonomous forex trading control surface powered by Gemini AI, MetaTrader 5, and automated risk controls.'
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <main className="app-shell">
          <header className="app-header">
            <div>
              <h1>Gemini Forex Autotrader</h1>
              <p className="subtitle">
                Fully automated MT5 execution with adaptive AI-driven risk
                management.
              </p>
            </div>
          </header>
          {children}
        </main>
      </body>
    </html>
  );
}
