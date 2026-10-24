import { useMemo } from 'react';
import { useLocation, useNavigate, useSearchParams as useRouterSearchParams } from 'react-router-dom';

export function usePathname(): string {
  return useLocation().pathname;
}

export function useSearchParams(): URLSearchParams {
  return useRouterSearchParams()[0];
}

export function useRouter() {
  const navigate = useNavigate();
  return useMemo(() => ({
    push: (href: string, options?: { scroll?: boolean }) => {
      void navigate(href);
      if (options?.scroll !== false) window.scrollTo(0, 0);
    },
    replace: (href: string, options?: { scroll?: boolean }) => {
      void navigate(href, { replace: true });
      if (options?.scroll !== false) window.scrollTo(0, 0);
    },
    back: () => { void navigate(-1); },
  }), [navigate]);
}
