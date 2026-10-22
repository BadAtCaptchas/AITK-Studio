'use client';

import YAML from 'yaml';
import Editor from '@monaco-editor/react';
import { useTheme } from './ThemeProvider';

import type { Job } from '@/types';

interface Props {
  job: Job;
}

const yamlConfig: YAML.DocumentOptions &
  YAML.SchemaOptions &
  YAML.ParseOptions &
  YAML.CreateNodeOptions &
  YAML.ToStringOptions = {
  indent: 2,
  lineWidth: 999999999999,
  defaultStringType: 'QUOTE_DOUBLE',
  defaultKeyType: 'PLAIN',
  directives: true,
};

function toYaml(obj: unknown): string {
  const doc = new YAML.Document(obj, yamlConfig);
  YAML.visit(doc, {
    Scalar(_key, node) {
      if (typeof node.value === 'string' && node.value.includes('\n')) {
        node.type = YAML.Scalar.BLOCK_LITERAL;
      }
    },
  });
  return doc.toString(yamlConfig);
}

export default function JobConfigViewer({ job }: Props) {
  const { theme } = useTheme();
  const editorValue = job?.job_config ? toYaml(JSON.parse(job.job_config)) : '';
  return (
    <>
      <Editor
        height="100%"
        width="100%"
        defaultLanguage="yaml"
        value={editorValue}
        theme={theme === 'dark' ? 'vs-dark' : 'light'}
        options={{
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          automaticLayout: true,
          readOnly: true,
        }}
      />
    </>
  );
}
