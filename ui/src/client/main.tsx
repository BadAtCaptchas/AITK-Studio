import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import RootLayout from '@/app/layout';
import HistoryNavigationGuard from '@/components/HistoryNavigationGuard';
import AppRoutes from './routes';
import PageBoundary from './PageBoundary';

const root = document.getElementById('root');
if (!root) throw new Error('Application root is missing');

createRoot(root).render(
  <PageBoundary>
    <HistoryNavigationGuard />
    <BrowserRouter useTransitions={false}>
      <RootLayout><AppRoutes /></RootLayout>
    </BrowserRouter>
  </PageBoundary>,
);
