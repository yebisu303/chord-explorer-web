"use strict";
// プレビュー用の小さなポリフォニックシンセ（AudioWorklet 版）。
// iOS 版の Packages/SynthCore/Sources/SynthCore/SynthCore.c を、計算の順序ごと移植したもの。
// ビルドを通さずにそのまま配信するので、import は使わない。
//
// - SynthCore: DSP の本体。ワークレットの外（Node のテスト）からも使える。
// - SynthProcessor: AudioWorkletProcessor。SynthCore を包んで、MessagePort とつなぐだけ。
//
// C 版は float（32 bit）で計算している。同じ波形にするため、演算 1 回ごとに Math.fround で
// 32 bit に丸めている（JS の double で計算してから丸めても、四則演算なら float の演算と同じ結果になる）。
// tests/synth.test.ts が、C 版の出力とサンプルごとに比べている。式を変えるときは、テストも見ること。

const F = Math.fround;

const VOICE_COUNT = 24;
const EVENT_CAPACITY = 32;
const MAX_CHORD_NOTES = 12;
const PRESET_COUNT = 6;

const PRESET_PIANO = 0; // ピアノ風（減衰するノコギリ波）
const PRESET_EPIANO = 1; // エレピ風（FM）
const PRESET_PAD = 2; // パッド（デチューンしたノコギリ波 2 本）
const PRESET_SAW = 3;
const PRESET_SQUARE = 4;
// 5 = サイン波

const EVENT_PLAY = 0;
const EVENT_STOP = 1;
const EVENT_PRESET = 2;

const STAGE_OFF = 0;
const STAGE_ATTACK = 1;
const STAGE_HOLD = 2;
const STAGE_RELEASE = 3;

const TWO_PI = F(6.283185307179586);
const NEG_TWO_PI = -TWO_PI;

// float にすると端数が出る定数（C の 0.45f など）。
const C_0_45 = F(0.45);
const C_0_7 = F(0.7);
const C_0_3 = F(0.3);
const C_0_35 = F(0.35);
const C_2_2 = F(2.2);
const C_1_007 = F(1.007);
const C_0_9965 = F(0.9965);
const C_0_137 = F(0.137);
const C_0_291 = F(0.291);
const C_6_9 = F(6.9);
const C_0_001 = F(0.001);

// 音色ごとの設定。C の PRESETS と同じ並び。
const PRESET_ATTACK = Float32Array.of(0.002, 0.002, 0.06, 0.004, 0.004, 0.006); // 秒
const PRESET_AMP_DECAY = Float32Array.of(1.4, 2.0, 0, 0, 0, 0); // 振幅が 1/e になる秒数。0 なら減衰しない
const PRESET_RELEASE = Float32Array.of(0.05, 0.05, 0.05, 0.05, 0.05, 0.05); // 離してから -60 dB までの秒数
const PRESET_COLOR_DECAY = Float32Array.of(0.35, 0.45, 0, 0, 0, 0);
const PRESET_GAIN = Float32Array.of(0.34, 0.3, 0.2, 0.2, 0.16, 0.3);

// 折り返しノイズを抑える補正（PolyBLEP）。
function polyBlep(t, dt) {
  if (t < dt) {
    t = F(t / dt);
    return F(F(F(t + t) - F(t * t)) - 1);
  }
  if (t > F(1 - dt)) {
    t = F(F(t - 1) / dt);
    return F(F(F(F(t * t) + t) + t) + 1);
  }
  return 0;
}

function saw(phase, dt) {
  return F(F(F(2 * phase) - 1) - polyBlep(phase, dt));
}

function square(phase, dt) {
  let value = phase < 0.5 ? 1 : -1;
  value = F(value + polyBlep(phase, dt));
  let shifted = F(phase + 0.5);
  if (shifted >= 1) shifted = F(shifted - 1);
  return F(value - polyBlep(shifted, dt));
}

function advance(phase, dt) {
  phase = F(phase + dt);
  if (phase >= 1) phase = F(phase - 1);
  return phase;
}

// 1 次ローパスの係数。
function lowpassGain(cutoff, sampleRate) {
  return F(1 - F(Math.exp(F(F(NEG_TWO_PI * cutoff) / sampleRate))));
}

class SynthCore {
  /** @param {number} sampleRate */
  constructor(sampleRate) {
    this.sampleRate = F(sampleRate);
    this.preset = PRESET_PIANO;

    // ボイスの状態。process の中でメモリを確保しないよう、型付き配列で持つ。
    this.stage = new Int32Array(VOICE_COUNT);
    this.voicePreset = new Int32Array(VOICE_COUNT);
    this.freq = new Float32Array(VOICE_COUNT);
    this.phaseA = new Float32Array(VOICE_COUNT); // 0...1
    this.phaseB = new Float32Array(VOICE_COUNT); // 0...1（2 本目のオシレータ / FM のモジュレータ）
    this.amp = new Float32Array(VOICE_COUNT); // エンベロープの現在値
    this.color = new Float32Array(VOICE_COUNT); // 音色用のエンベロープ。1 から減衰する
    this.lowpass = new Float32Array(VOICE_COUNT); // 1 次ローパスの状態
    this.age = new Float64Array(VOICE_COUNT); // 発音からのサンプル数
    this.hold = new Float64Array(VOICE_COUNT); // 自動で離すまでのサンプル数

    // サンプルレートと音色で決まる係数。C 版は毎サンプル計算しているが、値は変わらないので先に求めておく。
    const rate = this.sampleRate;
    this.attackStep = new Float32Array(PRESET_COUNT);
    this.holdFactor = new Float32Array(PRESET_COUNT);
    this.releaseFactor = new Float32Array(PRESET_COUNT);
    this.colorFactor = new Float32Array(PRESET_COUNT);
    for (let i = 0; i < PRESET_COUNT; i++) {
      this.attackStep[i] = F(1 / F(PRESET_ATTACK[i] * rate));
      this.holdFactor[i] = PRESET_AMP_DECAY[i] > 0 ? F(Math.exp(F(-1 / F(PRESET_AMP_DECAY[i] * rate)))) : 1;
      // 6.9 = ln(1000)。release 秒で -60 dB まで下がる。
      this.releaseFactor[i] = F(Math.exp(F(-C_6_9 / F(PRESET_RELEASE[i] * rate))));
      this.colorFactor[i] = PRESET_COLOR_DECAY[i] > 0 ? F(Math.exp(F(-1 / F(PRESET_COLOR_DECAY[i] * rate)))) : 1;
    }
    this.maxCutoff = F(C_0_45 * rate);
    this.gain1800 = lowpassGain(1800, rate);
    this.gain6000 = lowpassGain(6000, rate);

    // 発音・消音・音色のイベント。次の render の頭でまとめて処理する（C 版のリングと同じ容量、同じ順序）。
    this.eventType = new Int32Array(EVENT_CAPACITY);
    this.eventPreset = new Int32Array(EVENT_CAPACITY);
    this.eventCount = new Int32Array(EVENT_CAPACITY);
    this.eventSeconds = new Float32Array(EVENT_CAPACITY);
    this.eventNotes = new Uint8Array(EVENT_CAPACITY * MAX_CHORD_NOTES);
    this.eventWrite = 0;
    this.eventRead = 0;

    // 曲の再生。
    this.pending = null; // 次に鳴らす曲。total が 0 なら「止める」の合図
    this.current = null;
    this.position = -1; // UI に見せる再生位置。再生していなければ -1
    this.loop = false; // 終わりまで行ったら頭に戻る
    this.nextNote = 0;
    this.frame = 0;
    this.playing = false;
  }

  // MARK: イベントの受け付け（render の外から呼ぶ）

  /** @param {number} preset */
  setPreset(preset) {
    if (!(preset >= 0 && preset < PRESET_COUNT)) return;
    const slot = this._pushEvent(EVENT_PRESET);
    if (slot >= 0) this.eventPreset[slot] = preset;
  }

  /**
   * いま鳴っている音を止め、notes（MIDI ノート番号）を同時に鳴らす。seconds 後に自動で離す。
   * @param {ArrayLike<number>} notes
   * @param {number} seconds
   */
  play(notes, seconds) {
    if (!notes) return;
    const slot = this._pushEvent(EVENT_PLAY);
    if (slot < 0) return;
    const count = Math.min(notes.length, MAX_CHORD_NOTES);
    this.eventCount[slot] = count;
    this.eventSeconds[slot] = seconds;
    for (let i = 0; i < count; i++) {
      const note = notes[i] | 0;
      this.eventNotes[slot * MAX_CHORD_NOTES + i] = note < 0 ? 0 : note > 255 ? 255 : note;
    }
  }

  /** いま鳴っている音を離す。 */
  stop() {
    this._pushEvent(EVENT_STOP);
  }

  /**
   * 曲を頭から鳴らす。音は開始位置の小さい順に並べて渡す。位置と長さはサンプル数。
   * 再生中に呼ぶと、頭からやり直す。中身はコピーする。
   * @param {ArrayLike<number>} startFrames
   * @param {ArrayLike<number>} lengthFrames
   * @param {ArrayLike<number>} notes
   * @param {number} count
   * @param {number} totalFrames
   * @param {boolean} loop
   */
  sequenceStart(startFrames, lengthFrames, notes, count, totalFrames, loop) {
    if (!(count >= 0) || !(totalFrames > 0)) return;
    const sequence = {
      count,
      total: totalFrames,
      starts: new Float64Array(count),
      lengths: new Float64Array(count),
      notes: new Uint8Array(count),
    };
    for (let i = 0; i < count; i++) {
      sequence.starts[i] = startFrames[i];
      sequence.lengths[i] = lengthFrames[i];
      const note = notes[i] | 0;
      sequence.notes[i] = note < 0 ? 0 : note > 255 ? 255 : note;
    }
    this.loop = !!loop;
    // 位置を先に 0 にしておく（次の render までの間も「再生中」と読めるように）。
    this.position = 0;
    this.pending = sequence;
  }

  /** 終わりまで行ったら頭に戻るかどうかを、再生中でも切り替える。 */
  sequenceSetLoop(loop) {
    this.loop = !!loop;
  }

  /** 曲の再生を止める。 */
  sequenceStop() {
    this.pending = STOP_SEQUENCE;
  }

  /** いまの再生位置（頭からのサンプル数）。再生していなければ -1。 */
  sequencePosition() {
    return this.position;
  }

  // MARK: レンダリング

  /**
   * left / right に frames サンプルを書き込む（上書き）。right は null でもよい。
   * @param {Float32Array} left
   * @param {Float32Array | null} right
   * @param {number} frames
   */
  render(left, right, frames) {
    if (!left || frames <= 0) return;
    this._handleEvents();
    this._takeSequence();
    const wasPlaying = this.playing;
    const stage = this.stage;
    for (let frame = 0; frame < frames; frame++) {
      if (this.playing) this._advanceSequence();
      let mix = 0;
      for (let i = 0; i < VOICE_COUNT; i++) {
        if (stage[i] !== STAGE_OFF) mix = F(mix + this._renderVoice(i));
      }
      // 音数が多いときに割れないよう、なだらかに頭を抑える。
      mix = F(Math.tanh(mix));
      left[frame] = mix;
      if (right) right[frame] = mix;
    }
    // 位置を書くのは、再生中と、終わった瞬間だけ。止まっている間は書かない
    // （sequenceStart が書いた 0 を、-1 で上書きしないため）。
    if (this.playing) {
      this.position = this.frame;
    } else if (wasPlaying) {
      this.position = -1;
    }
  }

  // 空きがあればイベントの置き場所を返す。あふれたら捨てる（-1）。
  _pushEvent(type) {
    if (this.eventWrite - this.eventRead >= EVENT_CAPACITY) return -1;
    const slot = this.eventWrite % EVENT_CAPACITY;
    this.eventType[slot] = type;
    this.eventWrite++;
    return slot;
  }

  _handleEvents() {
    while (this.eventRead !== this.eventWrite) {
      const slot = this.eventRead % EVENT_CAPACITY;
      switch (this.eventType[slot]) {
        case EVENT_PRESET:
          this.preset = this.eventPreset[slot];
          break;
        case EVENT_STOP:
          this._releaseAll();
          break;
        case EVENT_PLAY: {
          this._releaseAll();
          const hold = Math.trunc(F(this.eventSeconds[slot] * this.sampleRate));
          const count = this.eventCount[slot];
          for (let i = 0; i < count; i++) {
            this._startVoice(this.eventNotes[slot * MAX_CHORD_NOTES + i], hold, i);
          }
          break;
        }
      }
      this.eventRead++;
    }
    // 番号が大きくなりすぎないよう、空になったら戻す。
    this.eventRead = 0;
    this.eventWrite = 0;
  }

  // 新しい曲が届いていたら切り替える。
  _takeSequence() {
    const incoming = this.pending;
    if (!incoming) return;
    this.pending = null;
    this.current = incoming;
    this.nextNote = 0;
    this.frame = 0;
    this.playing = incoming.total > 0;
    this._releaseAll();
    if (!this.playing) this.position = -1;
  }

  // 1 サンプル進める前に呼ぶ。いまの位置で始まる音を鳴らす。
  _advanceSequence() {
    const sequence = this.current;
    if (this.frame >= sequence.total) {
      if (!this.loop) {
        this.playing = false;
        return;
      }
      this.frame = 0;
      this.nextNote = 0;
    }
    while (this.nextNote < sequence.count && sequence.starts[this.nextNote] <= this.frame) {
      const index = this.nextNote;
      this._startVoice(sequence.notes[index], Math.trunc(sequence.lengths[index]), index);
      this.nextNote++;
    }
    this.frame++;
  }

  _releaseAll() {
    const stage = this.stage;
    for (let i = 0; i < VOICE_COUNT; i++) {
      if (stage[i] !== STAGE_OFF) stage[i] = STAGE_RELEASE;
    }
  }

  _allocateVoice() {
    const stage = this.stage;
    let best = 0;
    for (let i = 0; i < VOICE_COUNT; i++) {
      if (stage[i] === STAGE_OFF) return i;
      // 空きがなければ、離している中で一番小さい音、なければ一番古い音を奪う。
      const voiceReleasing = stage[i] === STAGE_RELEASE;
      const bestReleasing = stage[best] === STAGE_RELEASE;
      if (voiceReleasing !== bestReleasing) {
        if (voiceReleasing) best = i;
      } else if (voiceReleasing ? this.amp[i] < this.amp[best] : this.age[i] > this.age[best]) {
        best = i;
      }
    }
    return best;
  }

  _startVoice(note, holdFrames, index) {
    const i = this._allocateVoice();
    this.stage[i] = STAGE_ATTACK;
    this.voicePreset[i] = this.preset;
    this.freq[i] = F(440 * F(Math.pow(2, F(F(note - 69) / 12))));
    this.amp[i] = 0;
    this.color[i] = 1;
    this.lowpass[i] = 0;
    this.age[i] = 0;
    this.hold[i] = holdFrames;
    // 位相を少しずらして、同時発音の頭のピークをならす。
    const a = F(index * C_0_137);
    const b = F(index * C_0_291);
    this.phaseA[i] = F(a - Math.floor(a));
    this.phaseB[i] = F(b - Math.floor(b));
  }

  _renderVoice(i) {
    const rate = this.sampleRate;
    const preset = this.voicePreset[i];
    const freq = this.freq[i];
    const color = this.color[i];
    const phaseA = this.phaseA[i];
    let dt = F(freq / rate);
    if (dt > C_0_45) dt = C_0_45;
    let sample = 0;

    switch (preset) {
      case PRESET_PIANO: {
        // 弾いた瞬間は明るく、すぐにこもっていく。高い音ほど開く。
        let cutoff = F(F(freq * F(1.5 + F(9 * color))) + 200);
        if (cutoff > this.maxCutoff) cutoff = this.maxCutoff;
        const raw = F(F(C_0_7 * saw(phaseA, dt)) + F(C_0_3 * F(Math.sin(F(TWO_PI * phaseA)))));
        const state = this.lowpass[i];
        sample = F(state + F(lowpassGain(cutoff, rate) * F(raw - state)));
        this.lowpass[i] = sample;
        this.phaseA[i] = advance(phaseA, dt);
        break;
      }
      case PRESET_EPIANO: {
        // 2 オペレータの FM。深さが減衰して、丸い音に落ち着く。
        const phaseB = this.phaseB[i];
        const index = F(C_0_35 + F(C_2_2 * color));
        const mod = F(Math.sin(F(TWO_PI * phaseB)));
        sample = F(Math.sin(F(F(TWO_PI * phaseA) + F(index * mod))));
        this.phaseA[i] = advance(phaseA, dt);
        this.phaseB[i] = advance(phaseB, dt);
        break;
      }
      case PRESET_PAD: {
        const phaseB = this.phaseB[i];
        const dtA = F(dt * C_0_9965);
        const dtB = F(dt * C_1_007);
        const raw = F(0.5 * F(saw(phaseA, dtA) + saw(phaseB, dtB)));
        const state = this.lowpass[i];
        sample = F(state + F(this.gain1800 * F(raw - state)));
        this.lowpass[i] = sample;
        this.phaseA[i] = advance(phaseA, dtA);
        this.phaseB[i] = advance(phaseB, dtB);
        break;
      }
      case PRESET_SAW: {
        const state = this.lowpass[i];
        sample = F(state + F(this.gain6000 * F(saw(phaseA, dt) - state)));
        this.lowpass[i] = sample;
        this.phaseA[i] = advance(phaseA, dt);
        break;
      }
      case PRESET_SQUARE: {
        const state = this.lowpass[i];
        sample = F(state + F(this.gain6000 * F(square(phaseA, dt) - state)));
        this.lowpass[i] = sample;
        this.phaseA[i] = advance(phaseA, dt);
        break;
      }
      default:
        sample = F(Math.sin(F(TWO_PI * phaseA)));
        this.phaseA[i] = advance(phaseA, dt);
        break;
    }

    // エンベロープ
    let stage = this.stage[i];
    let amp = this.amp[i];
    switch (stage) {
      case STAGE_ATTACK:
        amp = F(amp + this.attackStep[preset]);
        if (amp >= 1) {
          amp = 1;
          stage = STAGE_HOLD;
        }
        break;
      case STAGE_HOLD:
        if (PRESET_AMP_DECAY[preset] > 0) amp = F(amp * this.holdFactor[preset]);
        break;
      case STAGE_RELEASE:
        amp = F(amp * this.releaseFactor[preset]);
        if (amp < C_0_001) stage = STAGE_OFF;
        break;
      default:
        break;
    }
    if (PRESET_COLOR_DECAY[preset] > 0) this.color[i] = F(color * this.colorFactor[preset]);

    const age = this.age[i] + 1;
    this.age[i] = age;
    if (stage !== STAGE_OFF && stage !== STAGE_RELEASE && age >= this.hold[i]) stage = STAGE_RELEASE;
    this.stage[i] = stage;
    this.amp[i] = amp;
    return F(F(sample * amp) * PRESET_GAIN[preset]);
  }
}

// 「曲を止める」の合図（C 版の total_frames == 0 の曲にあたる）。
const STOP_SEQUENCE = { count: 0, total: 0, starts: new Float64Array(0), lengths: new Float64Array(0), notes: new Uint8Array(0) };

// 再生位置をメインスレッドに知らせる間隔（秒）。間はメイン側が AudioContext の時刻から補う。
const POSITION_INTERVAL = 0.05;

// メインスレッド（src/synth/SynthPlayer.ts）から届くメッセージ:
//   { type: "preset", preset }
//   { type: "play", notes: number[], seconds }
//   { type: "stop" }
//   { type: "sequenceStart", id, starts, lengths, notes, total, loop }  位置と長さはサンプル数
//   { type: "sequenceLoop", loop }
//   { type: "sequenceStop" }
// メインスレッドへ送るメッセージ:
//   { type: "position", id, frame, time }
//     id の曲の再生位置が、AudioContext の時刻 time に frame（サンプル数）だった。再生が終わったら frame は -1。
class SynthProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.core = new SynthCore(sampleRate);
    this.sequenceId = 0;
    this.reportedPlaying = false;
    this.forceReport = false;
    this.framesSinceReport = 0;
    this.reportInterval = Math.round(sampleRate * POSITION_INTERVAL);
    // 送るたびにオブジェクトを作らないよう、使い回す。
    this.report = { type: "position", id: 0, frame: -1, time: 0 };
    this.port.onmessage = (event) => this.receive(event.data);
  }

  receive(message) {
    const core = this.core;
    switch (message.type) {
      case "preset":
        core.setPreset(message.preset);
        break;
      case "play":
        core.play(message.notes, message.seconds);
        break;
      case "stop":
        core.stop();
        break;
      case "sequenceStart":
        core.sequenceStart(message.starts, message.lengths, message.notes, message.notes.length, message.total, message.loop);
        this.sequenceId = message.id;
        this.forceReport = true;
        break;
      case "sequenceLoop":
        core.sequenceSetLoop(message.loop);
        break;
      case "sequenceStop":
        core.sequenceStop();
        break;
    }
  }

  process(inputs, outputs) {
    const output = outputs[0];
    if (!output || output.length === 0) return true;
    const left = output[0];
    const frames = left.length;
    const core = this.core;
    core.render(left, output.length > 1 ? output[1] : null, frames);
    for (let channel = 2; channel < output.length; channel++) output[channel].set(left);

    // 再生位置の報告。再生中は一定の間隔で、始まり・終わりはすぐに送る。
    const playing = core.playing;
    this.framesSinceReport += frames;
    if (this.forceReport || playing !== this.reportedPlaying || (playing && this.framesSinceReport >= this.reportInterval)) {
      if (playing || this.reportedPlaying || this.forceReport) {
        const report = this.report;
        report.id = this.sequenceId;
        report.frame = playing ? core.position : -1;
        report.time = currentTime + frames / sampleRate; // このブロックの終わりの時刻
        this.port.postMessage(report);
      }
      this.reportedPlaying = playing;
      this.forceReport = false;
      this.framesSinceReport = 0;
    }
    return true;
  }
}

if (typeof registerProcessor === "function") registerProcessor("chord-explorer-synth", SynthProcessor);
