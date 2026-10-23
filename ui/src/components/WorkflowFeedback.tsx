'use client';
import { createGlobalState } from 'react-global-hooks';
import { X } from 'lucide-react';

const feedback = createGlobalState<string | null>(null);
let feedbackOwner: symbol | null = null;
export function reportWorkflowError(message: string): () => void {
  const owner = Symbol();
  feedbackOwner = owner;
  feedback.set(message);
  // A recovered operation may dismiss only its own message, never a newer failure.
  return () => {
    if (feedbackOwner !== owner) return;
    feedbackOwner = null;
    feedback.set(null);
  };
}

export default function WorkflowFeedback() {
  const [message, setMessage] = feedback.use();
  if (!message) return null;
  return (
    <div
      role="alert"
      className="my-3 flex items-start gap-3 rounded-lg border border-rose-700 bg-gray-950 p-4 text-sm text-gray-100"
    >
      <p className="min-w-0 flex-1">{message}</p>
      <button type="button" aria-label="Dismiss message" onClick={() => setMessage(null)} className="shrink-0 p-1">
        <X size={16} />
      </button>
    </div>
  );
}
