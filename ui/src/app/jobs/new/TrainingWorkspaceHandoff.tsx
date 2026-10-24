import { useEffect } from 'react';
import { useRouter } from '@/navigation';
import { clearTrainingDraft, readTrainingDraft, restoreTrainingDraft } from '@/hooks/useTrainingDraft';
import { readStoredTrainingDraft, saveStoredTrainingDraft, type TrainingDraft } from '@/utils/trainingDraftStorage';
import { trainingWorkspaceDraftKey, trainingWorkspacePath, type TrainingWorkspace } from '@/domain/trainingWorkspaces';
import { clearConfigEditorSession, readConfigEditorSession, rememberConfigEditorSession } from '@/utils/configEditorSession';

export default function TrainingWorkspaceHandoff({ draft, workspace, query, baseline, sourceDraftKey, replaceDraft = false }: {
  draft: TrainingDraft; workspace: TrainingWorkspace; query: string; baseline?: string; sourceDraftKey?: string; replaceDraft?: boolean;
}) {
  const router = useRouter();
  useEffect(() => {
    const key = trainingWorkspaceDraftKey(workspace, new URLSearchParams(query));
    // Bookmarked links must not replace a newer draft already in this workspace.
    // An explicit config import does replace it, retaining its unsaved baseline.
    if (replaceDraft || (!readTrainingDraft(key) && !readStoredTrainingDraft(key))) {
      const stored = { draft, baseline: baseline ?? JSON.stringify(draft) };
      restoreTrainingDraft(key, stored);
      if (baseline) saveStoredTrainingDraft(key, stored);
      const editor = sourceDraftKey && !replaceDraft ? readConfigEditorSession(sourceDraftKey) : undefined;
      if (editor) rememberConfigEditorSession(key, editor);
    }
    if (sourceDraftKey && sourceDraftKey !== key) {
      clearTrainingDraft(sourceDraftKey);
      clearConfigEditorSession(sourceDraftKey);
    }
    router.replace(`${trainingWorkspacePath(workspace)}${query ? `?${query}` : ''}`);
  }, [draft, workspace, query, baseline, sourceDraftKey, replaceDraft, router]);
  return <p className="p-6 text-gray-400" role="status">Opening {workspace === 'continuation' ? 'Continue LoRA' : 'training setup'}…</p>;
}
