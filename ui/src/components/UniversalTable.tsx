import Loading from './Loading';
import classNames from 'classnames';
import { PageNotice } from '@/components/OperatorPrimitives';
import type React from 'react';

export interface TableColumn {
  title: string;
  key: string;
  render?: (row: any) => React.ReactNode;
  className?: string;
}

interface TableRow {
  [key: string]: any;
}

interface TableProps {
  columns: TableColumn[];
  rows: TableRow[];
  isLoading: boolean;
  theadClassName?: string;
  onRefresh: () => void;
  emptyTitle?: string;
  emptyDescription?: string;
  errorMessage?: string | null;
}

export default function UniversalTable({
  columns,
  rows,
  isLoading,
  theadClassName = 'text-gray-400',
  onRefresh = () => {},
  emptyTitle = 'No rows',
  emptyDescription = 'There is nothing to display for the current view.',
  errorMessage = null,
}: TableProps) {
  return (
    <div className="w-full overflow-hidden border border-gray-800 bg-gray-950/40">
      {isLoading ? (
        <div className="flex justify-center p-6">
          <Loading />
        </div>
      ) : errorMessage ? (
        <div className="p-3">
          <PageNotice
            tone="danger"
            title="Could not load table data"
            action={
              <button onClick={() => onRefresh()} className="operator-button py-1 text-xs">
                Retry
              </button>
            }
          >
            {errorMessage}
          </PageNotice>
        </div>
      ) : rows.length === 0 ? (
        <div className="p-3">
          <PageNotice
            tone="neutral"
            title={emptyTitle}
            action={
              <button onClick={() => onRefresh()} className="operator-button py-1 text-xs">
                Refresh
              </button>
            }
          >
            {emptyDescription}
          </PageNotice>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-left text-sm text-gray-300">
            <thead className={classNames('border-b border-gray-800 bg-gray-900 text-xs uppercase', theadClassName)}>
              <tr>
                {columns.map(column => (
                  <th key={column.key} className="px-3 py-2 font-medium">
                    {column.title}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows?.map((row, index) => {
                const rowClass = index % 2 === 0 ? 'bg-gray-950/20' : 'bg-gray-900/35';

                return (
                  <tr key={index} className={`${rowClass} border-b border-gray-800 last:border-b-0 hover:bg-gray-800/70`}>
                    {columns.map(column => (
                      <td key={column.key} className={classNames('px-3 py-2 align-middle', column.className)}>
                        {column.render ? column.render(row) : row[column.key]}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
