'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, CheckCircle2, Clipboard, ImagePlus, Loader2, ShieldCheck, Upload, XCircle } from 'lucide-react';
import { TopBar, MainContent } from '@/components/layout';
import { NumberInput, SelectInput, TextInput } from '@/components/formInputs';
import { PageNotice, ProgressBar } from '@/components/OperatorPrimitives';
import { apiClient } from '@/utils/api';
import {
  AUTHENLORA_BUILTIN_CODEC_BITS,
  AUTHENLORA_CODEC_OPTIONS,
  getAuthenloraCodecSelectValue,
} from '@/utils/authenloraCodecs';

type WatermarkCheckResult = {
  decoded_bits: string;
  msg_bits: number;
  confidence: number;
  threshold: number;
  codec: string;
  codec_path: string;
  codec_sha256: string;
  image: {
    width: number;
    height: number;
  };
  expected_secret_sha256: string | null;
  bit_accuracy: number | null;
  hamming_errors: number | null;
  match: boolean | null;
  zero_message: boolean;
  watermark_detected: boolean | null;
  watermark_status: 'not_detected' | 'verified' | 'mismatch' | 'candidate';
};

type NoticeTone = 'neutral' | 'info' | 'success' | 'warning' | 'danger';

const defaultCodec = 'builtin:authenlora_48bits';

function pct(value: number | null | undefined) {
  if (value == null || !Number.isFinite(value)) return '-';
  return `${(value * 100).toFixed(1)}%`;
}

function bitAccuracyText(value: number | null | undefined) {
  if (value == null) return 'Not checked';
  return pct(value);
}

function shortHash(value: string | null | undefined) {
  if (!value) return '-';
  return value.slice(0, 12);
}

function detectionLabel(status: WatermarkCheckResult['watermark_status']) {
  if (status === 'not_detected') return 'Not detected';
  if (status === 'verified') return 'Verified';
  if (status === 'mismatch') return 'No match';
  return 'Needs secret';
}

function bitAccuracyTone(result: WatermarkCheckResult): 'info' | 'success' | 'warning' | 'danger' {
  if (result.bit_accuracy == null) return result.watermark_status === 'not_detected' ? 'warning' : 'info';
  return result.match === true ? 'success' : 'danger';
}

export default function WatermarkCheckerPage() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [codecPath, setCodecPath] = useState(defaultCodec);
  const [msgBits, setMsgBits] = useState(48);
  const [expectedSecret, setExpectedSecret] = useState('');
  const [threshold, setThreshold] = useState(0.75);
  const [result, setResult] = useState<WatermarkCheckResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const selectedCodec = getAuthenloraCodecSelectValue(codecPath);
  const expectedSecretValid = !expectedSecret || (expectedSecret.length === msgBits && /^[01]+$/.test(expectedSecret));

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setPreviewUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);

  const resultTone = useMemo<NoticeTone>(() => {
    if (!result) return 'neutral';
    if (result.watermark_status === 'verified') return 'success';
    if (result.watermark_status === 'not_detected') return 'warning';
    if (result.watermark_status === 'mismatch') return 'danger';
    return 'info';
  }, [result]);

  const resultTitle = useMemo(() => {
    if (!result) return '';
    if (result.watermark_status === 'not_detected') return 'Watermark not detected';
    if (result.watermark_status === 'verified') return 'Watermark matches expected secret';
    if (result.watermark_status === 'mismatch') return 'Watermark does not match expected secret';
    return 'Candidate bits decoded';
  }, [result]);

  const handleCodecChange = (value: string) => {
    if (value === 'custom') {
      if (selectedCodec !== 'custom') setCodecPath('');
      return;
    }
    setCodecPath(value);
    const bits = AUTHENLORA_BUILTIN_CODEC_BITS[value];
    if (bits) setMsgBits(bits);
  };

  const handleFile = (nextFile: File | null) => {
    setFile(nextFile);
    setResult(null);
    setError(null);
    setCopied(false);
  };

  const runCheck = async () => {
    if (!file || loading) return;
    setLoading(true);
    setResult(null);
    setError(null);
    setCopied(false);

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('codec', codecPath);
      formData.append('msg_bits', String(msgBits));
      formData.append('threshold', String(threshold));
      if (expectedSecret.trim()) formData.append('expected_secret', expectedSecret.trim());

      const response = await apiClient.post('/api/watermark/check', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data as WatermarkCheckResult);
    } catch (err: unknown) {
      const maybeError = err as { response?: { data?: { error?: unknown } }; message?: unknown };
      const responseError = maybeError.response?.data?.error;
      setError(
        typeof responseError === 'string'
          ? responseError
          : typeof maybeError.message === 'string'
            ? maybeError.message
            : 'Watermark check failed.',
      );
    } finally {
      setLoading(false);
    }
  };

  const copyBits = async () => {
    if (!result?.decoded_bits) return;
    await navigator.clipboard.writeText(result.decoded_bits);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1500);
  };

  return (
    <>
      <TopBar className="h-14 border-gray-900 bg-[#02060a] px-4">
        <div className="flex min-w-0 flex-1 items-center gap-3">
          <ShieldCheck className="h-5 w-5 text-cyan-200" />
          <h1 className="truncate text-lg font-semibold text-gray-100">Watermark Checker</h1>
        </div>
      </TopBar>

      <MainContent className="operator-scrollbar-none bg-[#02060a] px-3 pt-16 text-gray-100 sm:px-4">
        <div className="mx-auto grid w-full max-w-6xl gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
          <section className="operator-panel overflow-hidden">
            <div className="operator-panel-header">
              <div className="flex min-w-0 items-center gap-2">
                <ImagePlus className="h-4 w-4 text-cyan-200" />
                <h2 className="truncate text-sm font-semibold text-gray-100">Image</h2>
              </div>
            </div>

            <div className="p-4">
              <input
                ref={inputRef}
                type="file"
                accept="image/png,image/jpeg,image/webp,image/jxl"
                className="hidden"
                onChange={event => handleFile(event.target.files?.[0] || null)}
              />

              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="flex min-h-[360px] w-full items-center justify-center border border-dashed border-gray-800 bg-gray-950/55 p-4 text-left transition-colors hover:border-cyan-900 hover:bg-gray-950"
              >
                {previewUrl ? (
                  <img src={previewUrl} alt={file?.name || 'Selected image'} className="max-h-[620px] max-w-full object-contain" />
                ) : (
                  <span className="flex flex-col items-center gap-3 text-center text-gray-400">
                    <Upload className="h-8 w-8 text-gray-500" />
                    <span className="text-sm font-medium text-gray-200">Select image</span>
                  </span>
                )}
              </button>

              {file ? (
                <div className="mt-3 flex min-w-0 flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
                  <span className="truncate">{file.name}</span>
                  <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                </div>
              ) : null}
            </div>
          </section>

          <aside className="space-y-4">
            <section className="operator-panel p-4">
              <div className="grid grid-cols-1 gap-3">
                <SelectInput label="Codec" value={selectedCodec} onChange={handleCodecChange} options={AUTHENLORA_CODEC_OPTIONS} />
                {selectedCodec === 'custom' ? (
                  <TextInput
                    label="Codec path"
                    value={codecPath}
                    onChange={setCodecPath}
                    placeholder="E:\\models\\authenlora_codec.pth"
                  />
                ) : null}
                <NumberInput label="Message bits" value={msgBits} onChange={value => setMsgBits(value ?? 48)} min={1} />
                <TextInput label="Expected secret" value={expectedSecret} onChange={setExpectedSecret} placeholder="optional binary secret" />
                <NumberInput label="Match threshold" value={threshold} onChange={value => setThreshold(value ?? 0.75)} min={0} max={1} />
              </div>

              {!expectedSecretValid ? (
                <PageNotice tone="warning" title="Expected secret length mismatch" className="mt-4">
                  Use a binary string with exactly {msgBits} bits.
                </PageNotice>
              ) : null}

              {error ? (
                <PageNotice tone="danger" title="Watermark check failed" className="mt-4">
                  {error}
                </PageNotice>
              ) : null}

              <button
                type="button"
                onClick={runCheck}
                disabled={!file || !codecPath.trim() || !expectedSecretValid || loading}
                className="operator-button mt-4 h-10 w-full justify-center border-emerald-800 bg-emerald-950/60 text-emerald-100 hover:bg-emerald-900 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <ShieldCheck className="h-4 w-4" />}
                Check watermark
              </button>
            </section>

            {result ? (
              <section className="operator-panel overflow-hidden">
                <div className="operator-panel-header">
                  <div className="flex min-w-0 items-center gap-2">
                    {result.watermark_status === 'verified' ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-300" />
                    ) : result.watermark_status === 'candidate' ? (
                      <ShieldCheck className="h-4 w-4 text-cyan-200" />
                    ) : result.watermark_status === 'not_detected' ? (
                      <AlertTriangle className="h-4 w-4 text-amber-300" />
                    ) : (
                      <XCircle className="h-4 w-4 text-rose-300" />
                    )}
                    <h2 className="truncate text-sm font-semibold text-gray-100">Result</h2>
                  </div>
                </div>
                <div className="space-y-4 p-4">
                  <PageNotice tone={resultTone} title={resultTitle}>
                    <div className="space-y-1">
                      <div>
                        {result.image.width}x{result.image.height} / {result.msg_bits} bits / codec {shortHash(result.codec_sha256)}
                      </div>
                      {result.watermark_status === 'not_detected' ? (
                        <div>The decoder returned the all-zero clean message, so this image is treated as unwatermarked.</div>
                      ) : result.watermark_status === 'candidate' ? (
                        <div>Enter the expected secret to verify a watermark match. Decoder confidence alone is not a presence signal.</div>
                      ) : null}
                    </div>
                  </PageNotice>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="border border-gray-900 bg-gray-950/60 p-3">
                      <div className="text-xs text-gray-500">Detection</div>
                      <div className="mt-1 font-semibold text-gray-100">{detectionLabel(result.watermark_status)}</div>
                      <div className="mt-2 text-[11px] text-gray-500">
                        {result.zero_message ? 'All-zero clean message' : result.watermark_detected === true ? 'Secret verified' : 'Secret required'}
                      </div>
                    </div>
                    <div className="border border-gray-900 bg-gray-950/60 p-3">
                      <div className="text-xs text-gray-500">Decoder confidence</div>
                      <div className="mt-1 font-semibold text-gray-100">{pct(result.confidence)}</div>
                      <ProgressBar value={result.confidence * 100} tone="info" className="mt-2" />
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="border border-gray-900 bg-gray-950/60 p-3">
                      <div className="text-xs text-gray-500">Bit accuracy</div>
                      <div className="mt-1 font-semibold text-gray-100">{bitAccuracyText(result.bit_accuracy)}</div>
                      <ProgressBar value={(result.bit_accuracy ?? 0) * 100} tone={bitAccuracyTone(result)} className="mt-2" />
                    </div>

                    <div className="border border-gray-900 bg-gray-950/60 p-3 text-sm">
                      <div className="text-xs text-gray-500">Bit errors</div>
                      <div className="mt-1 font-semibold text-gray-100">
                        {result.hamming_errors === null ? 'Not checked' : `${result.hamming_errors} / ${result.msg_bits}`}
                      </div>
                    </div>
                  </div>

                  <div>
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="text-xs font-semibold uppercase tracking-wide text-gray-500">Decoded bits</div>
                      <button type="button" className="operator-button px-2 py-1 text-xs" onClick={copyBits}>
                        <Clipboard className="h-3.5 w-3.5" />
                        {copied ? 'Copied' : 'Copy'}
                      </button>
                    </div>
                    <textarea
                      readOnly
                      value={result.decoded_bits}
                      className="h-28 w-full resize-none rounded-sm border border-gray-800 bg-gray-950 p-3 font-mono text-xs text-gray-200"
                    />
                  </div>
                </div>
              </section>
            ) : null}
          </aside>
        </div>
      </MainContent>
    </>
  );
}
