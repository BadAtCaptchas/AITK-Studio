import './globals.css';
import './studio-workspaces.css';
import './journey.css';
import './native-studio.css';
import { ThemeProvider } from '@/components/ThemeProvider';
import ConfirmModal from '@/components/ConfirmModal';
import AuthWrapper from '@/components/AuthWrapper';
import DocModal from '@/components/DocModal';
import { CaptionDatasetModal } from '@/components/CaptionDatasetModal';
import MergeLoRAsModal from '@/components/MergeLoRAsModal';
import AppShell from '@/components/AppShell';

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const authRequired = window.aitkConfig.authRequired;

  return (
    <>
        <ThemeProvider>
          <AuthWrapper authRequired={authRequired}>
            <AppShell>{children}</AppShell>
          </AuthWrapper>
        </ThemeProvider>
        <ConfirmModal />
        <DocModal />
        <CaptionDatasetModal />
        <MergeLoRAsModal />
    </>
  );
}
