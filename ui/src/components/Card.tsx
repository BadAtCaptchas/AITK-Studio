import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react';
import { FaChevronDown } from 'react-icons/fa';
import classNames from 'classnames';

interface CardProps {
  title?: string;
  children?: React.ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
}

const Card: React.FC<CardProps> = ({ title, children, collapsible, defaultOpen }) => {
  if (collapsible) {
    return (
      <Disclosure as="section" className="operator-panel" defaultOpen={defaultOpen}>
        {({ open }) => (
          <>
            <DisclosureButton className="operator-panel-header w-full text-left">
              <div className="flex-1">
                {title && (
                  <h2 className={classNames('text-xs font-semibold uppercase tracking-wide text-gray-400', { 'mb-0': !open })}>
                    {title}
                  </h2>
                )}
              </div>
              <FaChevronDown className={`ml-2 inline-block transition-transform ${open ? 'rotate-180' : ''}`} />
            </DisclosureButton>
            <DisclosurePanel className="px-3 pb-3 pt-2">{children ?? null}</DisclosurePanel>
          </>
        )}
      </Disclosure>
    );
  }
  return (
    <section className="operator-panel">
      {title && (
        <div className="operator-panel-header">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-gray-400">{title}</h2>
        </div>
      )}
      <div className="space-y-2 px-3 pb-3 pt-2">{children ?? null}</div>
    </section>
  );
};

export default Card;
