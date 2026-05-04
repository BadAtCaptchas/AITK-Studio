import classNames from 'classnames';

interface Props {
  className?: string;
  children?: React.ReactNode;
}

export const TopBar: React.FC<Props> = ({ children, className }) => {
  return (
    <div
      className={classNames(
        'absolute top-0 left-0 w-full h-14 bg-black/95 border-b border-white/10 shadow-sm z-10 flex items-center px-4',
        className,
      )}
    >
      {children ? children : null}
    </div>
  );
};

export const MainContent: React.FC<Props> = ({ children, className }) => {
  return (
    <div className={classNames('pt-16 px-4 absolute top-0 left-0 w-full h-full overflow-auto bg-black', className)}>
      {children ? children : null}
    </div>
  );
};
