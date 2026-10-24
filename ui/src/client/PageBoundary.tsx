import { Component, type ReactNode } from 'react';
import GlobalError from '@/app/error';

export default class PageBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) return <GlobalError error={this.state.error} reset={() => window.location.reload()} />;
    return this.props.children;
  }
}
