/* tslint:disable */
/* eslint-disable */

/** The brain's options, each optional. */
export interface BrainOptions {
  /** The depth of every neuron's history (default 256). */
  h?: number
  /** The neighborhood policy: "eventsToActions" (default) or "allOn". */
  policy?: string
  /** What an action dimension outputs when every candidate is negative: "leastBad" (default) or "nothing". */
  actionPolicy?: string
  /** The dispatch: regions, and columns per region (one of each by default). */
  regions?: number
  columns?: number
  /** Fixed for the life of the brain (default true); a brain built with it off is loaded from a backup to be tested. */
  learning?: boolean
}
/** One neuron dimension of a channel: its name and how many values it takes. */
export interface DimSpec {
  name: string
  resolution: number
}
/** A channel: its event dimension, its action dimension, and the spatial axes it is laid out over, by index. */
export interface ChannelSpec {
  name: string
  event?: DimSpec
  action?: DimSpec
  axes?: Array<number>
}
/** What declaring a channel returns: its id, and the ids of its event and action dimensions. */
export interface ChannelIds {
  channelId: number
  eventDim?: number
  actionDim?: number
}
/** A declared channel: its id, name, event and action dimensions, and the spatial axes it is laid out over. */
export interface ChannelOut {
  channelId: number
  name: string
  eventDim?: number
  actionDim?: number
  axes: Array<number>
}
/** How many activations stand on the apex in the last frame, by kind. */
export interface ApexCounts {
  calls: number
  events: number
  actions: number
}
/** One symbol the world reports in a frame: its dimension, its value, and its position on the spatial axes. */
export interface InputSpec {
  dim: number
  bucket: number
  position?: Array<number>
}
/** A reward: its amount, the action channels it pays (all when absent), and its span of frames (the longest when absent). */
export interface RewardSpec {
  reward: number
  channels?: Array<number>
  frames?: number
}
/** One output for the frame ahead: a dimension's winning value at a position. */
export interface WinnerOut {
  dim: number
  bucket: number
  position: Array<number>
  kind: string
  estimate: number
  share: number
  voters: number
}
/** The counts of the last frame. */
export interface StatsOut {
  frame: number
  levels: number
  apexActivations: number
  functionsAdded: number
  functionsRetired: number
  voters: number
  neurons: number
  functions: number
  connections: number
}
/** A neuron's fixed facts. */
export interface NeuronOut {
  id: number
  kind: string
  dim?: number
  bucket: number
  channel?: number
  alive: boolean
}
/** A relation of the dictionary: two poses and the offset from the first to the second (time first). */
export interface RelationOut {
  from: number
  to: number
  offset: Array<number>
}
/** A function of the dictionary. */
export interface FunctionOut {
  id: number
  description: string
  nodes: Array<number>
  relations: Array<RelationOut>
  interneuron?: number
  isCycle: boolean
  origin?: number
}
/** A pose that names a call: its interneuron, and its phase from one node to another over a number of laps. */
export interface PoseOut {
  neuron: number
  from: number
  to: number
  laps: number
}
/** An activation's place on the board: the frame it is recorded under and its index there. */
export interface ActRefOut {
  frame: number
  index: number
}
/** A relation an activation's neuron returned at it: the other activation, its offset, and its strength. */
export interface BoardRelationOut {
  other: ActRefOut
  offset: Array<number>
  strength: number
}
/**
 * An open activation on the board: `frame` is the frame it is recorded under, `time` the frame it stands at, and
 * `reach` its reach in time first, then on each spatial axis.
 */
export interface ActivationOut {
  index: number
  neuron: number
  pose: number
  kind: string
  frame: number
  time: number
  position: Array<number>
  reach: Array<number>
  strong: boolean
  onApex: boolean
  gone: boolean
  level: number
  coveredAt?: number
  covers: Array<ActRefOut>
  coverers: Array<ActRefOut>
  firedBy?: number
  relations: Array<BoardRelationOut>
}
/** A frame still open on the board, with its activations in order. */
export interface FrameOut {
  frame: number
  activations: Array<ActivationOut>
}
export declare class Brain {
  /** Create a brain from its options. */
  constructor(options?: BrainOptions | undefined | null)
  /** Declare a spatial axis; returns its index (1 or 2). */
  declareAxis(name: string): number
  /** Declare a channel; its base neurons are made in declaration order. */
  declareChannel(spec: ChannelSpec): ChannelIds
  /** The spatial axes declared, by name, in index order from 1. */
  listAxes(): Array<string>
  /** The channels declared, in id order. */
  listChannels(): Array<ChannelOut>
  /** Process one frame: what was reported and what ran, with the rewards; returns the output for the frame ahead. */
  processFrame(inputs: Array<InputSpec>, rewards: Array<RewardSpec>): Array<WinnerOut>
  /** Clear every open activation; keep the dictionary and the connections. */
  resetContext(): void
  /** The counts of the last frame. */
  getStats(): StatsOut
  /** The frame number of the last frame. */
  getFrameNumber(): number
  /** How many neurons are alive. */
  countLivingNeurons(): number
  /** How many neurons were ever allotted, living or not. */
  countNeurons(): number
  /** The longest reach in time among the activations still open. */
  getLongestReach(): number
  /** The base neuron of a dimension's value. */
  findBaseNeuron(dim: number, bucket: number): number
  /** The fixed facts of the named neurons, in the order named. */
  readNeurons(ids: Array<number>): Array<NeuronOut>
  /** Describe the named neurons: kind, rows and connections, the functions each is the interneuron of, and its first sixteen connections. */
  describeNeurons(ids: Array<number>): Array<string>
  /** How many living neurons stand at each height. */
  countHeights(): Array<number>
  /** How many functions of the dictionary never took a call. */
  countFunctionsNeverCalled(): number
  /** Every function of the dictionary, in id order. */
  getDictionary(): Array<FunctionOut>
  /**
   * A fingerprint of the dictionary: a hash of every function's description, sorted.
   * Two fingerprints differ where a function started, left, or reads differently.
   */
  readDictionaryFingerprint(): string
  /**
   * Each function's id and a stamp of what it is made of (its nodes, its relations and its interneuron), in id order,
   * as one flat list: id, stamp, id, stamp. A stamp changes where what the function is made of changes.
   */
  readFunctionStamps(): Array<number>
  /** The named functions of the dictionary, in the order named; a function no longer there is left out. */
  describeFunctions(ids: Array<number>): Array<FunctionOut>
  /** How many functions the dictionary holds. */
  countFunctions(): number
  /** A pose as the dictionary writes it. */
  describePose(pose: number): string
  /** The call a pose names, if it names one: its interneuron and its phase. */
  readPose(pose: number): PoseOut | null
  /** The functions an interneuron holds, in order. */
  listFunctionsOf(interneuron: number): Array<number>
  /** A relation as the dictionary holds it: from `from` to `to` at the offset, or turned round where the dictionary keeps it the other way. */
  canonRelation(from: number, to: number, offset: Array<number>): RelationOut
  /** The function that holds a relation, as the dictionary holds it; none where no function holds it. */
  findHolder(relation: RelationOut): number | null
  /** Every frame still open on the board, oldest first; each activation's relations only when `withRelations` is true. */
  getBoard(withRelations?: boolean): Array<FrameOut>
  /** How many activations stand on the apex in the last frame, by kind; none where the board is empty. */
  countApex(): ApexCounts
  /** The last frame on the board; each activation's relations only when `withRelations` is true. */
  getLastFrame(withRelations?: boolean): FrameOut | null
  /** Save the brain to `<jobDir>/backups/<label>/`; returns the folder written. */
  save(jobDir: string, label: string): string
  /** Load `<jobDir>/backups/<label>/` into a brain that has not run. */
  load(jobDir: string, label: string): void
  /** Save the board to `<jobDir>/contexts/<label>/`; returns the folder written. */
  saveContext(jobDir: string, label: string): string
  /** Load `<jobDir>/contexts/<label>/` into a brain loaded from the backup saved with it. */
  loadContext(jobDir: string, label: string): void
}
