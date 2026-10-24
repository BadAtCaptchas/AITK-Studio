import { Link as RouterLink, type LinkProps } from 'react-router-dom';
import type { Ref } from 'react';

type Props = Omit<LinkProps, 'to'> & { href: string; ref?: Ref<HTMLAnchorElement> };

export default function Link({ href, ...props }: Props) {
  return <RouterLink to={href} {...props} />;
}
