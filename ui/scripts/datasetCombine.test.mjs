import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { test } from 'node:test';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const {
  combineDatasets,
  datasetCombineRequestHasKeyMaterial,
  hasPlaintextEncryptedOutputFiles,
} = require('../dist/src/server/datasetCombine.js');
const { ENCRYPTED_DATASET_MANIFEST } = require('../dist/src/server/encryptedDatasets.js');

const CATALOG_AAD = 'aitk-encrypted-catalog:v1';
const AUTH_TAG_BYTES = 16;

test('datasetCombineRequestHasKeyMaterial detects raw keys before remote forwarding', () => {
  assert.equal(datasetCombineRequestHasKeyMaterial(null), false);
  assert.equal(datasetCombineRequestHasKeyMaterial({ sourceDatasets: ['plain'], outputName: 'combined' }), false);
  assert.equal(
    datasetCombineRequestHasKeyMaterial({
      sourceDatasets: ['locked'],
      outputName: 'combined',
      encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: 'source-key' }],
    }),
    true,
  );
  assert.equal(
    datasetCombineRequestHasKeyMaterial({
      sourceDatasets: ['plain'],
      outputName: 'combined',
      outputEncrypted: true,
      outputKeyB64: 'output-key',
    }),
    true,
  );
});

test('combining layered documents into encrypted output fails without publishing an incomplete dataset', async t => {
  const root = await makeRoot();
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const id = 'document123';
  const folder = await writePlainDataset(root, 'layered', [{ path: 'art.png', caption: 'poster' }, { path: `.layers/${id}/0.png` }]);
  await fs.writeFile(path.join(folder, '.layers', id, 'manifest.json'), JSON.stringify({ format: 'aitk.layered-image', version: 1, id, composite: 'art.png', caption: 'art.txt', width: 64, height: 64, order: 'bottom-to-top', layers: [{ path: `.layers/${id}/0.png`, name: 'target' }] }));
  const key = crypto.randomBytes(32);
  await writePlainDataset(root, 'plain', [{ path: 'plain.png', caption: 'plain' }]);
  await assert.rejects(combineDatasets(root, { sourceDatasets: ['layered', 'plain'], outputName: 'encrypted', outputEncrypted: true, outputEncryptedManifest: manifestForKey(key), outputKeyB64: b64(key) }), /ENCRYPTED_DATASET_MIGRATION_REQUIRED/);
  assert.equal(fsSync.existsSync(path.join(root, 'encrypted')), false);
  assert.equal(fsSync.existsSync(path.join(folder, 'art.png')), true);
});

function b64(value) {
  return Buffer.from(value).toString('base64');
}

function encryptPayload(key, data, aad) {
  const nonce = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv('aes-256-gcm', key, nonce, { authTagLength: AUTH_TAG_BYTES });
  cipher.setAAD(Buffer.from(aad, 'utf8'));
  const encrypted = Buffer.concat([cipher.update(Buffer.from(data)), cipher.final(), cipher.getAuthTag()]);
  return {
    nonce: b64(nonce),
    data: b64(encrypted),
  };
}

function decryptPayload(key, payload, aad) {
  const encrypted = Buffer.from(payload.data, 'base64');
  const ciphertext = encrypted.subarray(0, encrypted.length - AUTH_TAG_BYTES);
  const tag = encrypted.subarray(encrypted.length - AUTH_TAG_BYTES);
  const decipher = crypto.createDecipheriv('aes-256-gcm', key, Buffer.from(payload.nonce, 'base64'), {
    authTagLength: AUTH_TAG_BYTES,
  });
  decipher.setAAD(Buffer.from(aad, 'utf8'));
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(ciphertext), decipher.final()]);
}

function manifestForKey(key, catalog = { version: 1, items: [] }) {
  return {
    format: 'aitk-encrypted-dataset',
    version: 1,
    crypto: {
      algorithm: 'AES-256-GCM',
      kdf: {
        type: 'KEYFILE-SHA256',
        keyLength: 32,
      },
    },
    catalog: encryptPayload(key, Buffer.from(JSON.stringify(catalog), 'utf8'), CATALOG_AAD),
  };
}

async function makeRoot() {
  return fs.mkdtemp(path.join(os.tmpdir(), 'aitk-combine-test-'));
}

async function writePlainDataset(root, name, files) {
  const folder = path.join(root, name);
  await fs.mkdir(folder, { recursive: true });
  for (const file of files) {
    const filePath = path.join(folder, ...file.path.split('/'));
    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, file.data ?? `media:${file.path}`);
    if (file.caption !== undefined) {
      const parsed = path.parse(filePath);
      await fs.writeFile(path.join(parsed.dir, `${parsed.name}.txt`), file.caption);
    }
  }
  return folder;
}

async function writeEncryptedDataset(root, name, key, specs) {
  const folder = path.join(root, name);
  await fs.mkdir(path.join(folder, 'objects'), { recursive: true });
  const items = [];

  for (const spec of specs) {
    const id = spec.id || crypto.randomBytes(8).toString('hex');
    const objectPath = `objects/${id}.bin`;
    await fs.writeFile(
      path.join(folder, objectPath),
      JSON.stringify(encryptPayload(key, Buffer.from(spec.data), `aitk-encrypted-object:${objectPath}`)),
    );

    const item = {
      id,
      name: spec.name,
      extension: path.extname(spec.name).toLowerCase(),
      mimeType: 'image/png',
      mediaKind: 'image',
      objectPath,
      size: Buffer.byteLength(spec.data),
      width: 4,
      height: 3,
      createdAt: '2026-05-30T00:00:00.000Z',
      updatedAt: '2026-05-30T00:00:00.000Z',
    };

    if (spec.caption !== undefined) {
      const captionObjectPath = `objects/${id}.caption.bin`;
      await fs.writeFile(
        path.join(folder, captionObjectPath),
        JSON.stringify(
          encryptPayload(key, Buffer.from(spec.caption), `aitk-encrypted-object:${captionObjectPath}`),
        ),
      );
      item.captionObjectPath = captionObjectPath;
    }
    items.push(item);
  }

  await fs.writeFile(
    path.join(folder, ENCRYPTED_DATASET_MANIFEST),
    JSON.stringify(manifestForKey(key, { version: 1, items }), null, 2),
  );
  return folder;
}

async function readOutputCatalog(datasetFolder, key) {
  const manifest = JSON.parse(await fs.readFile(path.join(datasetFolder, ENCRYPTED_DATASET_MANIFEST), 'utf8'));
  return JSON.parse(decryptPayload(key, manifest.catalog, CATALOG_AAD).toString('utf8'));
}

test('combineDatasets combines plain datasets and renames duplicate stems', async () => {
  const root = await makeRoot();
  await writePlainDataset(root, 'alpha', [
    { path: 'image.png', data: 'alpha-image', caption: 'alpha caption' },
    { path: 'same.jpg', data: 'alpha-same', caption: 'same alpha' },
  ]);
  await writePlainDataset(root, 'beta', [
    { path: 'nested/image.png', data: 'beta-image', caption: 'beta caption' },
    { path: 'same.png', data: 'beta-same', caption: 'same beta' },
  ]);
  await fs.mkdir(path.join(root, 'combined'));

  const result = await combineDatasets(root, {
    sourceDatasets: ['alpha', 'beta'],
    outputName: 'combined',
  });

  assert.equal(result.dataset.name, 'combined_2');
  const output = path.join(root, result.dataset.name);
  assert.deepEqual((await fs.readdir(output)).sort(), [
    'image.png',
    'image.txt',
    'image_2.png',
    'image_2.txt',
    'same.jpg',
    'same.txt',
    'same_2.png',
    'same_2.txt',
  ]);
  assert.equal(await fs.readFile(path.join(output, 'image.txt'), 'utf8'), 'alpha caption');
  assert.equal(await fs.readFile(path.join(output, 'image_2.txt'), 'utf8'), 'beta caption');
  assert.equal(await fs.readFile(path.join(output, 'same_2.txt'), 'utf8'), 'same beta');
});


const { decryptDataset } = require('../dist/src/server/datasetDecryption.js');
test('remote migration requires HTTPS and capability before forwarding credentials', async () => {
  const ts = require('typescript');
  const code = ts.transpileModule(await fs.readFile(new URL('../src/app/api/datasets/decrypt/route.ts', import.meta.url), 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText;
  let baseURL = 'http://worker.invalid', supported = false;
  const requests = [];
  const remote = {
    isLocalWorker: id => id === 'local',
    getRemoteWorker: async () => ({ base_url: baseURL }),
    withoutRemoteRedirects: init => ({ ...init, redirect: 'manual' }),
    remoteJson: async (_worker, route, init) => {
      requests.push({ route, init });
      return route === '/api/capabilities' ? { datasetDecryption: supported ? 1 : 0 } : { id: 'migration-id' };
    },
  };
  const loadedModule = { exports: {} };
  new Function('require', 'module', 'exports', code)(name => name === '@/server/remoteClient' ? remote : name.startsWith('@/') ? require('../dist/src/' + name.slice(2) + '.js') : require(name), loadedModule, loadedModule.exports);
  const payload = { datasetName: 'locked', outputName: 'plain', worker_id: 'worker', keyB64: b64(crypto.randomBytes(32)) };
  const start = () => loadedModule.exports.POST(new Request('https://studio.invalid/api/datasets/decrypt', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }));
  assert.equal((await start()).status, 400); assert.equal(requests.length, 0);
  baseURL = 'https://worker.invalid';
  const incompatible = await start();
  assert.equal(incompatible.status, 409); assert.match((await incompatible.json()).error, /Upgrade/);
  assert.equal(requests.length, 1); assert.equal(requests[0].init.body, undefined);
  supported = true;
  assert.equal((await start()).status, 202);
  assert.equal(requests.at(-1).route, '/api/datasets/decrypt');
  assert.equal(JSON.parse(requests.at(-1).init.body).keyB64, payload.keyB64);
  assert.ok(requests.every(request => request.init.redirect === 'manual'));
});
test('password, key file and WebAuthn PRF credentials unlock a complete migration', async t => {
  const { unlockEncryptedDatasetKey } = require('../dist/src/utils/encryptedDatasets.js');
  const { encryptWithWebAuthnPrfKey, webAuthnPrfWrappedKeyAad } = require('../dist/src/utils/webauthnPrfCrypto.js');
  for (const provider of ['password', 'keyFile', 'webauthnPrf']) await t.test(provider, async t => {
    const root = await makeRoot(); t.after(() => fs.rm(root, { recursive: true, force: true }));
    let key = crypto.randomBytes(32), kdf, credential;
    if (provider === 'password') {
      const salt = crypto.randomBytes(16);
      key = crypto.pbkdf2Sync('migration-password', salt, 600000, 32, 'sha256');
      kdf = { type: 'PBKDF2-SHA256', salt: b64(salt), iterations: 600000, keyLength: 32 };
      credential = { provider, password: 'migration-password' };
    } else if (provider === 'keyFile') {
      const file = new File(['original-key-file'], 'key.bin');
      key = crypto.createHash('sha256').update(Buffer.from(await file.arrayBuffer())).digest();
      kdf = { type: 'KEYFILE-SHA256', keyLength: 32 };
      credential = { provider, file };
    } else {
      const prf = crypto.randomBytes(32), rawID = crypto.randomBytes(24), id = rawID.toString('base64url'), saltB64 = b64(crypto.randomBytes(32));
      const wrappedKey = await encryptWithWebAuthnPrfKey(prf, key, webAuthnPrfWrappedKeyAad('localhost', id, saltB64));
      kdf = { type: 'WEBAUTHN-PRF', keyLength: 32, rpId: 'localhost', credentials: [{ id, saltB64, transports: ['usb'], createdAt: new Date().toISOString(), wrappedKey: { algorithm: 'AES-256-GCM', ...wrappedKey } }] };
      credential = { provider };
      for (const [name, value] of Object.entries({ window: { isSecureContext: true }, navigator: { credentials: { get: async () => ({ type: 'public-key', id, rawId: rawID, getClientExtensionResults: () => ({ prf: { results: { first: prf } } }) }) } }, PublicKeyCredential: class {} })) {
        const prior = Object.getOwnPropertyDescriptor(globalThis, name);
        Object.defineProperty(globalThis, name, { value, configurable: true });
        t.after(() => { if (prior) Object.defineProperty(globalThis, name, prior); else delete globalThis[name]; });
      }
    }
    const source = await writeEncryptedDataset(root, 'locked', key, [{ name: 'image.png', data: 'unchanged-media', caption: 'unchanged-caption' }]);
    const manifest = JSON.parse(await fs.readFile(path.join(source, ENCRYPTED_DATASET_MANIFEST), 'utf8'));
    manifest.crypto.kdf = kdf;
    await fs.writeFile(path.join(source, ENCRYPTED_DATASET_MANIFEST), JSON.stringify(manifest));
    const unlocked = await unlockEncryptedDatasetKey(manifest, credential);
    assert.equal(unlocked.rawKeyB64, b64(key));
    const result = await decryptDataset(migrationOptions(root, key, { keyB64: unlocked.rawKeyB64 }));
    assert.equal(await fs.readFile(path.join(root, result.datasetName, 'image.png'), 'utf8'), 'unchanged-media');
    assert.equal(await fs.readFile(path.join(root, result.datasetName, 'image.txt'), 'utf8'), 'unchanged-caption');
    assert.deepEqual(JSON.parse(await fs.readFile(path.join(source, ENCRYPTED_DATASET_MANIFEST), 'utf8')), manifest);
  });
});
function migrationOptions(root, key, extra = {}) {
  return { root, sourceName: 'locked', outputName: 'locked_decrypted', keyB64: b64(key), operationID: crypto.randomBytes(32).toString('hex'), progress: async () => {}, beforeCommit: async () => {}, ...extra };
}
test('legacy encryption is rejected by combine, even with a valid key', async t => {
  const root = await makeRoot(); t.after(() => fs.rm(root, { recursive: true, force: true }));
  const key = crypto.randomBytes(32);
  await writePlainDataset(root, 'plain', [{ path: 'x.png' }]);
  await writeEncryptedDataset(root, 'locked', key, [{ name: 'x.png', data: 'secret' }]);
  for (const request of [
    { sourceDatasets: ['plain', 'locked'], outputName: 'combined', encryptedDatasetKeys: [{ datasetName: 'locked', keyB64: b64(key) }] },
    { sourceDatasets: ['plain'], outputName: 'combined', outputEncrypted: true },
  ]) await assert.rejects(combineDatasets(root, request), /ENCRYPTED_DATASET_MIGRATION_REQUIRED/);
  assert.equal(fsSync.existsSync(path.join(root, 'combined')), false);
});
test('migration preserves media, caption bytes, root captions and resolves name collisions', async t => {
  const root = await makeRoot(); t.after(() => fs.rm(root, { recursive: true, force: true }));
  const key = crypto.randomBytes(32);
  const source = await writeEncryptedDataset(root, 'locked', key, [
    { name: 'same.png', data: 'original1', caption: 'caption\r\nbytes\u0000' },
    { name: 'same.jpg', data: 'original2', caption: '' },
  ]);
  const catalog = await readOutputCatalog(source, key);
  catalog.rootCaption = 'root\r\ncaption';
  await fs.writeFile(path.join(source, ENCRYPTED_DATASET_MANIFEST), JSON.stringify(manifestForKey(key, catalog)));
  const originalManifest = await fs.readFile(path.join(source, ENCRYPTED_DATASET_MANIFEST));
  const originalObject = await fs.readFile(path.join(source, catalog.items[0].objectPath));
  const options = migrationOptions(root, key);
  const result = await decryptDataset(options);
  assert.equal(result.itemCount, 2); assert.equal(result.captionCount, 2);
  assert.deepEqual(result.renamed, [{ original: 'same.jpg', name: 'same_2.jpg' }]);
  const dest = path.join(root, result.datasetName);
  assert.equal(await fs.readFile(path.join(dest, 'same.png'), 'utf8'), 'original1');
  assert.equal(await fs.readFile(path.join(dest, 'same.txt'), 'utf8'), 'caption\r\nbytes\u0000');
  assert.equal(await fs.readFile(path.join(dest, 'same_2.txt'), 'utf8'), '');
  assert.equal(await fs.readFile(path.join(dest, 'ROOT_CAPTION.txt'), 'utf8'), catalog.rootCaption);
  assert.deepEqual(await fs.readFile(path.join(source, ENCRYPTED_DATASET_MANIFEST)), originalManifest);
  assert.deepEqual(await fs.readFile(path.join(source, catalog.items[0].objectPath)), originalObject);
  assert.deepEqual(await decryptDataset(options), result, 'published retries are idempotent');
  await assert.rejects(decryptDataset(migrationOptions(root, key)), /already exists/);
});
test('failed, canceled and racing migrations retain all originals and clean owned staging', async t => {
  for (const mode of ['wrong-key', 'corrupt', 'missing', 'traversal', 'unsafe-name', 'cancel', 'race']) {
    await t.test(mode, async t => {
      const root = await makeRoot(); t.after(() => fs.rm(root, { recursive: true, force: true }));
      const key = crypto.randomBytes(32);
      const source = await writeEncryptedDataset(root, 'locked', key, [{ id: 'media1', name: 'test.png', data: 'media', caption: 'caption' }]);
      const options = migrationOptions(root, key);
      if (mode === 'wrong-key') options.keyB64 = b64(crypto.randomBytes(32));
      if (mode === 'corrupt') await fs.writeFile(path.join(source, 'objects/media1.bin'), '{"nonce":"bad","data":"bad"}');
      if (mode === 'missing') await fs.unlink(path.join(source, 'objects/media1.bin'));
      if (mode === 'traversal' || mode === 'unsafe-name') {
        const catalog = await readOutputCatalog(source, key);
        if (mode === 'traversal') catalog.items[0].objectPath = '../outside.bin';
        else catalog.items[0].name = '../escape.png';
        await fs.writeFile(path.join(source, ENCRYPTED_DATASET_MANIFEST), JSON.stringify(manifestForKey(key, catalog)));
      }
      if (mode === 'cancel') options.progress = async () => { throw new Error('Operation canceled'); };
      if (mode === 'race') options.beforeCommit = async () => { await fs.mkdir(path.join(root, options.outputName)); await fs.writeFile(path.join(root, options.outputName, 'keep'), 'existing'); };
      const original = await fs.readFile(path.join(source, ENCRYPTED_DATASET_MANIFEST));
      await assert.rejects(decryptDataset(options));
      assert.deepEqual(await fs.readFile(path.join(source, ENCRYPTED_DATASET_MANIFEST)), original);
      assert.equal((await fs.readdir(root)).some(name => name.startsWith('.aitk-decrypt-')), false);
      if (mode === 'race') assert.equal(await fs.readFile(path.join(root, options.outputName, 'keep'), 'utf8'), 'existing');
      else assert.equal(fsSync.existsSync(path.join(root, options.outputName)), false);
    });
  }
});
test('migration rejects objects symlinked outside the encrypted dataset', async t => {
  const root = await makeRoot(); t.after(() => fs.rm(root, { recursive: true, force: true }));
  const key = crypto.randomBytes(32);
  const source = await writeEncryptedDataset(root, 'locked', key, [{ id: 'media', name: 'test.png', data: 'media' }]);
  const object = path.join(source, 'objects/media.bin');
  const outside = path.join(root, 'outside.bin');
  await fs.rename(object, outside);
  try { await fs.symlink(outside, object); } catch (error) { if (error.code === 'EPERM') { t.skip('File symlink privilege unavailable'); return; } throw error; }
  await assert.rejects(decryptDataset(migrationOptions(root, key)), /escapes/);
  assert.equal(fsSync.existsSync(outside), true);
});
