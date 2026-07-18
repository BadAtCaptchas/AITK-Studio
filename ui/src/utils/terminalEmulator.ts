// Stateful plain-text terminal rendering for incremental job-log chunks.
const MAX_LINES = 5000;
const MAX_PENDING = 4096;

export class TerminalEmulator {
  private lines: string[] = [''];
  private row = 0;
  private col = 0;
  private pending = '';

  reset(): void {
    this.lines = [''];
    this.row = 0;
    this.col = 0;
    this.pending = '';
  }

  write(chunk: string): void {
    const text = this.pending + chunk;
    this.pending = '';
    let index = 0;
    while (index < text.length) {
      const code = text.charCodeAt(index);
      if (code === 0x1b) {
        const consumed = this.consumeEscape(text, index);
        if (consumed === 0) {
          if (text.length - index <= MAX_PENDING) {
            this.pending = text.slice(index);
          }
          return;
        }
        index += consumed;
      } else if (code === 0x0d) {
        this.col = 0;
        index += 1;
      } else if (code === 0x0a) {
        this.newline();
        index += 1;
      } else if (code === 0x08) {
        if (this.col > 0) this.col -= 1;
        index += 1;
      } else if (code === 0x09) {
        this.putText(' '.repeat(8 - (this.col % 8)));
        index += 1;
      } else if (code < 0x20 || code === 0x7f) {
        index += 1;
      } else {
        let end = index + 1;
        while (end < text.length) {
          const next = text.charCodeAt(end);
          if (next < 0x20 || next === 0x7f) break;
          end += 1;
        }
        this.putText(text.slice(index, end));
        index = end;
      }
    }
  }

  toLines(): string[] {
    return this.lines.slice();
  }

  toString(): string {
    return this.lines.join('\n');
  }

  private putText(value: string): void {
    let line = this.lines[this.row];
    if (line.length < this.col) line = line.padEnd(this.col);
    this.lines[this.row] =
      line.slice(0, this.col) + value + line.slice(this.col + value.length);
    this.col += value.length;
  }

  private newline(): void {
    this.row += 1;
    this.col = 0;
    this.ensureRow();
    const excess = this.lines.length - MAX_LINES;
    if (excess > 0) {
      this.lines.splice(0, excess);
      this.row = Math.max(0, this.row - excess);
    }
  }

  private ensureRow(): void {
    while (this.lines.length <= this.row) this.lines.push('');
  }

  private consumeEscape(text: string, index: number): number {
    if (index + 1 >= text.length) return 0;
    const kind = text[index + 1];

    if (kind === '[') {
      let end = index + 2;
      while (end < text.length && /[0-9;?]/.test(text[end])) end += 1;
      while (end < text.length && text[end] >= ' ' && text[end] <= '/') end += 1;
      if (end >= text.length) return 0;
      const final = text[end];
      if (final >= '@' && final <= '~') {
        this.applyCsi(text.slice(index + 2, end), final);
        return end - index + 1;
      }
      return 2;
    }

    if (kind === ']') {
      for (let end = index + 2; end < text.length; end += 1) {
        if (text.charCodeAt(end) === 0x07) return end - index + 1;
        if (text.charCodeAt(end) === 0x1b) {
          if (end + 1 >= text.length) return 0;
          if (text[end + 1] === '\\') return end - index + 2;
        }
      }
      return 0;
    }

    if (kind === 'M') {
      this.row = Math.max(0, this.row - 1);
    }
    return 2;
  }

  private applyCsi(params: string, final: string): void {
    const args = params
      .replace(/^\?/, '')
      .split(';')
      .map(value => Number.parseInt(value, 10));
    const arg = (index: number, fallback: number): number => {
      const value = args[index];
      return Number.isNaN(value) || value === undefined ? fallback : value;
    };

    switch (final) {
      case 'A':
      case 'F':
        this.row = Math.max(0, this.row - arg(0, 1));
        if (final === 'F') this.col = 0;
        break;
      case 'B':
      case 'e':
      case 'E':
        this.row += arg(0, 1);
        this.ensureRow();
        if (final === 'E') this.col = 0;
        break;
      case 'C':
      case 'a':
        this.col += arg(0, 1);
        break;
      case 'D':
        this.col = Math.max(0, this.col - arg(0, 1));
        break;
      case 'G':
      case '`':
        this.col = Math.max(0, arg(0, 1) - 1);
        break;
      case 'K': {
        const mode = arg(0, 0);
        const line = this.lines[this.row];
        if (mode === 0) {
          this.lines[this.row] = line.slice(0, this.col);
        } else if (mode === 1) {
          this.lines[this.row] =
            ' '.repeat(Math.min(this.col + 1, line.length)) +
            line.slice(this.col + 1);
        } else if (mode === 2) {
          this.lines[this.row] = '';
        }
        break;
      }
      case 'J': {
        const mode = arg(0, 0);
        if (mode === 0) {
          this.lines[this.row] = this.lines[this.row].slice(0, this.col);
          this.lines.length = this.row + 1;
        } else if (mode === 2 || mode === 3) {
          this.reset();
        }
        break;
      }
      default:
        break;
    }
  }
}
