import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Forge Control Center",
  description: "Operator dashboard for the Forge autonomous engineering runtime"
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
