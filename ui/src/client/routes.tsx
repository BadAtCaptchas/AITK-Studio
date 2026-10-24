import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Loading from '@/app/loading';
import NotFound from '@/app/not-found';
import PageBoundary from './PageBoundary';

const pages = [
  ['/dashboard', lazy(() => import('@/app/dashboard/page'))],
  ['/datasets', lazy(() => import('@/app/datasets/page'))],
  ['/datasets/:datasetName', lazy(() => import('@/app/datasets/[datasetName]/page'))],
  ['/generate', lazy(() => import('@/app/generate/page'))],
  ['/generate/live', lazy(() => import('@/app/generate/live/page'))],
  ['/jobs', lazy(() => import('@/app/jobs/page'))],
  ['/jobs/new', lazy(() => import('@/app/jobs/new/page'))],
  ['/jobs/secure-remote-captioning', lazy(() => import('@/app/jobs/secure-remote-captioning/page'))],
  ['/jobs/:jobID', lazy(() => import('@/app/jobs/[jobID]/page'))],
  ['/models', lazy(() => import('@/app/models/page'))],
  ['/settings', lazy(() => import('@/app/settings/page'))],
  ['/tools', lazy(() => import('@/app/tools/page'))],
  ['/updates', lazy(() => import('@/app/updates/page'))],
  ['/watermark', lazy(() => import('@/app/watermark/page'))],
  ['/workflows', lazy(() => import('@/app/workflows/page'))],
  ['/workflows/ideogram', lazy(() => import('@/app/workflows/ideogram/page'))],
] as const;

export default function AppRoutes() {
  const location = useLocation();
  return <PageBoundary key={location.pathname}>
    <Suspense fallback={<Loading />}>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        {pages.map(([path, Page]) => <Route key={path} path={path} element={<Page />} />)}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Suspense>
  </PageBoundary>;
}
