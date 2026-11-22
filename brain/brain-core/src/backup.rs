/// Backup / restore for brain state.
///
/// A backup is a folder of CSV files, no header rows, at `<job_dir>/backups/<label>/`:
///   - `brain.csv`         key,value — the history depth, the policies, the frame number, the counts, the reaches
///   - `act_dims.csv`      name,radius — the activation dimensions, time first
///   - `dims.csv`          id,name,channel,kind,resolution,first
///   - `channels.csv`      id,name,event,action,axes
///   - `neurons.csv`       id,kind,dim,bucket,channel,axes,alive,h,now_frame — the registry, and per neuron the
///                         latest frame a row was written for
///   - `rows.csv`          neuron,frame,index,own,neighbors — each neuron's history, oldest first
///   - `connections.csv`   neuron,own,target,offset,strength,earnings — each neuron's connections, in its order
///   - `functions.csv`     id,joined,inter,nodes,relations,at — the dictionary
///   - `retired.csv`       the same, with the frame until which each is kept readable (R38)
///   - `poses.csv`         neuron,from,to,laps,named — the pose table, in order
///   - `interneurons.csv`  neuron,functions — each interneuron's functions, in order
///   - `ledger.csv`        neuron,due — the ledger, in order, each with the frame it waits for
///   - `last_func.csv`, `reach_of.csv` — per interneuron, the function that fired it last, and its reach
/// The relation index is not written: it is a count of the rows, recounted on load.
///
/// A context is the board and what the machine keeps about it, at `<job_dir>/contexts/<label>/`:
///   - `context.csv`       key,value — the run's first frame
///   - `frames.csv`        frame,actions run,rewards
///   - `acts.csv`          one open activation per line, in its frame's order
///   - `pending_weak.csv`  what the selection named for the frame ahead (R40)
///   - `edges.csv`         set,edge — the covered, lifted, refused and pre-standing relations (D24, D55)
///   - `edge_lists.csv`    map,edge,edges — what each edge stands for, and what each relation was lifted to
///   - `act_lists.csv`     map,activation,activations — covered relations, what each call took, what it read through
///   - `act_edges.csv`     call,edges — the board relations each call's fit covered
///   - `kinds.csv`, `grown.csv`, `covered_by.csv` — the kind each activation decided, the laps, what each call covered
///
/// Within a field a list is space-separated and its items' parts colon-separated; an empty list or a missing value is
/// written `-`.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::board::*;
use crate::brain::{Brain, FrameStats};
use crate::dictionary::{Dictionary, Func, FuncId, Rel};
use crate::neuron::{Conn, Nb, Row};
use crate::thalamus::NeuronFacts;
use crate::types::*;

/// How many timestamped backups `latest` keeps.
const MAX_BACKUPS: usize = 10;

// ── Folders ─────────────────────────────────────────────────────────────────

/// The folder to write: a timestamped one for `latest`, else the named one, emptied first.
fn make_folder(root: &Path, label: &str) -> Result<PathBuf, String> {
    fs::create_dir_all(root).map_err(|e| format!("Failed to create {}: {}", root.display(), e))?;
    let folder = if label == "latest" { root.join(format_timestamp()) } else { root.join(label) };
    if label != "latest" && folder.exists() {
        fs::remove_dir_all(&folder).map_err(|e| format!("Failed to clear {}: {}", folder.display(), e))?;
    }
    fs::create_dir_all(&folder).map_err(|e| format!("Failed to create {}: {}", folder.display(), e))?;
    Ok(folder)
}

/// The folder to read: the newest timestamped one for `latest`, else the named one.
fn find_folder(root: &Path, label: &str) -> Result<PathBuf, String> {
    if label != "latest" {
        let named = root.join(label);
        if !named.exists() { return Err(format!("no backup {}: {} does not exist", label, named.display())); }
        return Ok(named);
    }
    let mut stamped = list_timestamp_folders(root);
    stamped.sort();
    stamped.last().map(|n| root.join(n)).ok_or_else(|| format!("no timestamped backup in {}", root.display()))
}

/// Remove the oldest timestamped folders beyond the ones kept.
fn prune_folders(root: &Path) {
    let mut stamped = list_timestamp_folders(root);
    stamped.sort();
    let stale = stamped.len().saturating_sub(MAX_BACKUPS);
    for name in stamped.iter().take(stale) { let _ = fs::remove_dir_all(root.join(name)); }
}

fn list_timestamp_folders(root: &Path) -> Vec<String> {
    let Ok(entries) = fs::read_dir(root) else { return Vec::new() };
    entries.filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| is_timestamp_folder(n))
        .collect()
}

/// Check if a folder name matches the timestamp format `YYYY-MM-DD_HH-mm-ss`.
fn is_timestamp_folder(name: &str) -> bool {
    if name.len() != 19 { return false; }
    let b = name.as_bytes();
    b[4] == b'-' && b[7] == b'-' && b[10] == b'_' && b[13] == b'-' && b[16] == b'-'
        && b.iter().enumerate().filter(|&(i, _)| ![4, 7, 10, 13, 16].contains(&i)).all(|(_, c)| c.is_ascii_digit())
}

/// Build a `YYYY-MM-DD_HH-mm-ss` timestamp in UTC.
fn format_timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
    let (days, tod) = (secs / 86400, secs % 86400);
    // the civil date from the day count (Howard Hinnant's algorithm)
    let z = days as i64 + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{:04}-{:02}-{:02}_{:02}-{:02}-{:02}", y, m, d, tod / 3600, (tod % 3600) / 60, tod % 60)
}

// ── The codec ───────────────────────────────────────────────────────────────

/// Write one table, a line per row, the fields comma-separated.
fn write_table(folder: &Path, name: &str, rows: impl Iterator<Item = Vec<String>>) -> Result<(), String> {
    let path = folder.join(name);
    let file = File::create(&path).map_err(|e| format!("Failed to create {}: {}", path.display(), e))?;
    let mut w = BufWriter::new(file);
    for row in rows {
        for (i, field) in row.iter().enumerate() {
            debug_assert!(!field.contains(',') && !field.contains('\n'), "a field holds no comma or newline");
            if i > 0 { w.write_all(b",").map_err(|e| e.to_string())?; }
            w.write_all(field.as_bytes()).map_err(|e| e.to_string())?;
        }
        w.write_all(b"\n").map_err(|e| e.to_string())?;
    }
    w.flush().map_err(|e| format!("Failed to write {}: {}", path.display(), e))
}

/// Read one table into its rows of fields.
fn read_table(folder: &Path, name: &str) -> Result<Vec<Vec<String>>, String> {
    let path = folder.join(name);
    let text = fs::read_to_string(&path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    Ok(text.lines().filter(|l| !l.is_empty()).map(|l| l.split(',').map(|f| f.to_string()).collect()).collect())
}

fn bad(what: &str) -> String { format!("backup: cannot read {}", what) }

fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> { s.parse::<T>().map_err(|_| bad(s)) }

fn write_list<T>(v: &[T], f: impl Fn(&T) -> String) -> String { if v.is_empty() { "-".into() } else { v.iter().map(f).collect::<Vec<_>>().join(" ") } }

fn read_list<T>(s: &str, f: impl Fn(&str) -> Result<T, String>) -> Result<Vec<T>, String> { if s == "-" { Ok(Vec::new()) } else { s.split(' ').map(f).collect() } }

fn write_opt<T>(v: Option<T>, f: impl Fn(T) -> String) -> String { v.map_or("-".into(), f) }

fn read_opt<T>(s: &str, f: impl Fn(&str) -> Result<T, String>) -> Result<Option<T>, String> { if s == "-" { Ok(None) } else { f(s).map(Some) } }

fn write_bool(b: bool) -> String { if b { "1".into() } else { "0".into() } }

fn read_bool(s: &str) -> Result<bool, String> { match s { "1" => Ok(true), "0" => Ok(false), _ => Err(bad(s)) } }

fn write_kind(k: Kind) -> String { (k as u8).to_string() }

fn read_kind(s: &str) -> Result<Kind, String> { match s { "0" => Ok(Kind::Event), "1" => Ok(Kind::Action), "2" => Ok(Kind::Inter), _ => Err(bad(s)) } }

fn write_pos(p: &Pos) -> String { (0..p.len()).map(|d| p[d].to_string()).collect::<Vec<_>>().join(":") }

fn write_axes(a: &Axes) -> String { if a.space().is_empty() { "-".into() } else { a.space().iter().map(|d| d.to_string()).collect::<Vec<_>>().join(":") } }

fn read_axes(s: &str) -> Result<Axes, String> {
    if s == "-" { return Ok(Axes::time()); }
    let v: Vec<usize> = s.split(':').map(parse).collect::<Result<_, _>>()?;
    Ok(Axes::from_dims(&v))
}

fn read_pos(s: &str) -> Result<Pos, String> {
    let v: Vec<i32> = s.split(':').map(parse).collect::<Result<_, _>>()?;
    Ok(Pos::from_slice(&v))
}

fn write_act(r: &ActRef) -> String { format!("{}:{}", r.frame, r.index) }

fn read_act(s: &str) -> Result<ActRef, String> {
    let (f, i) = s.split_once(':').ok_or_else(|| bad(s))?;
    Ok(ActRef { frame: parse(f)?, index: parse(i)? })
}

fn write_edge(e: &(ActRef, ActRef)) -> String { format!("{}:{}", write_act(&e.0), write_act(&e.1)) }

fn read_edge(s: &str) -> Result<(ActRef, ActRef), String> {
    let p: Vec<&str> = s.split(':').collect();
    if p.len() != 4 { return Err(bad(s)); }
    Ok((ActRef { frame: parse(p[0])?, index: parse(p[1])? }, ActRef { frame: parse(p[2])?, index: parse(p[3])? }))
}

fn write_rel(r: &Rel) -> String { format!("{}:{}:{}", r.a, r.b, write_pos(&r.off)) }

fn read_rel(s: &str) -> Result<Rel, String> {
    let (a, rest) = s.split_once(':').ok_or_else(|| bad(s))?;
    let (b, off) = rest.split_once(':').ok_or_else(|| bad(s))?;
    Ok(Rel { a: parse(a)?, b: parse(b)?, off: read_pos(off)? })
}

fn write_func(f: &Func) -> Vec<String> {
    vec![f.id.to_string(), f.joined.to_string(), write_opt(f.inter, |n| n.to_string()), write_list(&f.nodes, |s| s.to_string()),
        write_list(&f.rels, write_rel), write_list(&f.at, |a| write_opt(*a, |s| s.to_string()))]
}

fn read_func(r: &[String]) -> Result<Func, String> {
    Ok(Func {
        id: parse(&r[0])?,
        joined: parse(&r[1])?,
        inter: read_opt(&r[2], parse)?,
        nodes: read_list(&r[3], parse)?,
        rels: read_list(&r[4], read_rel)?,
        at: read_list(&r[5], |s| read_opt(s, parse))?,
    })
}

/// A table of key-value pairs.
fn read_pairs(folder: &Path, name: &str) -> Result<FxHashMap<String, String>, String> {
    Ok(read_table(folder, name)?.into_iter().filter(|r| r.len() == 2).map(|r| (r[0].clone(), r[1].clone())).collect())
}

fn read_key<T: std::str::FromStr>(pairs: &FxHashMap<String, String>, key: &str) -> Result<T, String> {
    parse(pairs.get(key).ok_or_else(|| bad(key))?)
}

fn write_policy(p: &[[bool; 3]; 3]) -> String { p.iter().flatten().map(|b| if *b { '1' } else { '0' }).collect() }

fn read_policy(s: &str) -> Result<[[bool; 3]; 3], String> {
    let b: Vec<bool> = s.chars().map(|c| c == '1').collect();
    if b.len() != 9 { return Err(bad(s)); }
    Ok([[b[0], b[1], b[2]], [b[3], b[4], b[5]], [b[6], b[7], b[8]]])
}

// ── The brain ───────────────────────────────────────────────────────────────

impl Brain {
    /// Save the brain's state to `<job_dir>/backups/<label>/`: `latest` writes a timestamped folder and keeps the
    /// newest few, any other label writes that folder anew.
    pub fn save(&mut self, job_dir: &Path, label: &str) -> Result<PathBuf, String> {
        let root = job_dir.join("backups");
        let folder = make_folder(&root, label)?;
        self.write_brain(&folder)?;
        if label == "latest" { prune_folders(&root); }
        Ok(folder)
    }

    /// Load a backup from `<job_dir>/backups/<label>/` into a brain that has not run: its regions, columns and
    /// learning stay as it was built.
    pub fn load(&mut self, job_dir: &Path, label: &str) -> Result<(), String> {
        if self.frame_no != 0 || self.thalamus.registry.len() != 0 { return Err("a backup loads only into a brain that has not run".into()); }
        let folder = find_folder(&job_dir.join("backups"), label)?;
        self.read_brain(&folder)
    }

    /// Save the board and what the machine keeps about it to `<job_dir>/contexts/<label>/`.
    pub fn save_context(&self, job_dir: &Path, label: &str) -> Result<PathBuf, String> {
        let root = job_dir.join("contexts");
        let folder = make_folder(&root, label)?;
        self.write_context(&folder)?;
        if label == "latest" { prune_folders(&root); }
        Ok(folder)
    }

    /// Load a context from `<job_dir>/contexts/<label>/`, into a brain loaded from the backup saved with it.
    pub fn load_context(&mut self, job_dir: &Path, label: &str) -> Result<(), String> {
        let folder = find_folder(&job_dir.join("contexts"), label)?;
        self.read_context(&folder)
    }

    fn write_brain(&mut self, folder: &Path) -> Result<(), String> {
        let p = &self.cfg.policy;
        let scalars: Vec<(&str, String)> = vec![
            ("h", self.cfg.h.to_string()),
            ("past", write_policy(&p.past)),
            ("future", write_policy(&p.future)),
            ("action", match p.action { ActionPolicy::LeastBad => "least_bad".into(), ActionPolicy::Nothing => "nothing".into() }),
            ("frame_no", self.frame_no.to_string()),
            ("alive", self.alive.to_string()),
            ("base_count", self.base_count.to_string()),
            ("max_reach_t", self.max_reach_t.to_string()),
            ("span_s", self.span_s.to_string()),
            ("next_function", self.thalamus.dict.next_id.to_string()),
            ("next_join", self.thalamus.dict.next_join.to_string()),
        ];
        write_table(folder, "brain.csv", scalars.into_iter().map(|(k, v)| vec![k.to_string(), v]))?;
        write_table(folder, "act_dims.csv", self.act_dims.iter().map(|d| vec![d.name.clone(), d.radius.to_string()]))?;
        write_table(folder, "dims.csv", self.dims.iter().map(|d| vec![d.id.to_string(), d.name.clone(), d.channel.to_string(), write_kind(d.kind), d.resolution.to_string(), d.first.to_string()]))?;
        write_table(folder, "channels.csv", self.channels.iter().map(|c| vec![c.id.to_string(), c.name.clone(), write_opt(c.event, |d| d.to_string()), write_opt(c.action, |d| d.to_string()), write_axes(&c.axes)]))?;
        // the neurons: the registry, and what each holds, read in one batch
        let ids: Vec<NeuronId> = (0..self.thalamus.registry.len() as NeuronId).collect();
        let held = self.thalamus.export_neurons(ids.clone());
        let reg = &self.thalamus.registry;
        write_table(folder, "neurons.csv", ids.iter().zip(&held).map(|(&id, (h, now, _, _))| {
            let f = &reg[id];
            vec![id.to_string(), write_kind(f.kind), write_opt(f.dim, |d| d.to_string()), f.bucket.to_string(), write_opt(f.channel, |c| c.to_string()), write_axes(&f.axes), write_bool(f.alive), h.to_string(), now.to_string()]
        }))?;
        write_table(folder, "rows.csv", ids.iter().zip(&held).flat_map(|(&id, (_, _, rows, _))| rows.iter().map(move |r| vec![
            id.to_string(), r.act.frame.to_string(), r.act.index.to_string(), r.own.to_string(),
            write_list(&r.nbs, |nb| format!("{}:{}:{}", write_act(&nb.act), nb.sym, write_pos(&nb.off))),
        ])))?;
        write_table(folder, "connections.csv", ids.iter().zip(&held).flat_map(|(&id, (_, _, _, conns))| conns.iter().map(move |((own, t, off), c)| vec![
            id.to_string(), own.to_string(), t.to_string(), write_pos(off), c.strength.to_string(), c.earnings.to_string(),
        ])))?;
        // the dictionary, the functions kept readable, and the pose table
        let dict = &self.thalamus.dict;
        let mut fids: Vec<&FuncId> = dict.funcs.keys().collect();
        fids.sort();
        write_table(folder, "functions.csv", fids.iter().map(|id| write_func(&dict.funcs[*id])))?;
        let mut rids: Vec<&FuncId> = self.retired_funcs.keys().collect();
        rids.sort();
        write_table(folder, "retired.csv", rids.iter().map(|id| { let (f, until) = &self.retired_funcs[*id]; let mut r = write_func(f); r.push(until.to_string()); r }))?;
        write_table(folder, "poses.csv", dict.syms.iter().map(|(n, p)| vec![n.to_string(), p.from.to_string(), p.to.to_string(), p.laps.to_string(), write_bool(dict.sym_index.contains_key(&(*n, *p)))]))?;
        let mut inters: Vec<(&NeuronId, &Vec<FuncId>)> = dict.by_inter.iter().collect();
        inters.sort();
        write_table(folder, "interneurons.csv", inters.into_iter().map(|(n, fs)| vec![n.to_string(), write_list(fs, |f| f.to_string())]))?;
        // the ledger, and per interneuron the function that fired it last and its reach
        write_table(folder, "ledger.csv", self.ledger.iter().map(|n| vec![n.to_string(), write_opt(self.ledger_due.get(n), |d| d.to_string())]))?;
        let mut waits: Vec<(&NeuronId, &FrameNumber)> = self.ledger_due.iter().filter(|(n, _)| !self.ledger.contains(n)).collect();
        waits.sort();
        write_table(folder, "ledger_due.csv", waits.into_iter().map(|(n, d)| vec![n.to_string(), d.to_string()]))?;
        let mut last: Vec<(&NeuronId, &FuncId)> = self.last_func.iter().collect();
        last.sort();
        write_table(folder, "last_func.csv", last.into_iter().map(|(n, f)| vec![n.to_string(), f.to_string()]))?;
        let mut reach: Vec<(&NeuronId, &Pos)> = self.reach_of.iter().collect();
        reach.sort();
        write_table(folder, "reach_of.csv", reach.into_iter().map(|(n, p)| vec![n.to_string(), write_pos(p)]))
    }

    fn read_brain(&mut self, folder: &Path) -> Result<(), String> {
        let s = read_pairs(folder, "brain.csv")?;
        self.cfg.h = read_key(&s, "h")?;
        self.cfg.policy.past = read_policy(s.get("past").ok_or_else(|| bad("past"))?)?;
        self.cfg.policy.future = read_policy(s.get("future").ok_or_else(|| bad("future"))?)?;
        self.cfg.policy.action = match s.get("action").map(|x| x.as_str()) { Some("nothing") => ActionPolicy::Nothing, Some("least_bad") => ActionPolicy::LeastBad, _ => return Err(bad("action")) };
        self.frame_no = read_key(&s, "frame_no")?;
        self.alive = read_key(&s, "alive")?;
        self.base_count = read_key(&s, "base_count")?;
        self.max_reach_t = read_key(&s, "max_reach_t")?;
        self.span_s = read_key(&s, "span_s")?;
        self.act_dims = read_table(folder, "act_dims.csv")?.iter().map(|r| Ok(ActDim { name: r[0].clone(), radius: parse(&r[1])? })).collect::<Result<_, String>>()?;
        self.thalamus.dict.dims = self.act_dims.len();
        self.dims = read_table(folder, "dims.csv")?.iter().map(|r| Ok(NeuronDim { id: parse(&r[0])?, name: r[1].clone(), channel: parse(&r[2])?, kind: read_kind(&r[3])?, resolution: parse(&r[4])?, first: parse(&r[5])? })).collect::<Result<_, String>>()?;
        self.channels = read_table(folder, "channels.csv")?.iter().map(|r| Ok(Channel { id: parse(&r[0])?, name: r[1].clone(), event: read_opt(&r[2], parse)?, action: read_opt(&r[3], parse)?, axes: read_axes(&r[4])? })).collect::<Result<_, String>>()?;
        // the neurons: each with its facts, its rows and its connections, created in one batch
        let mut neurons: Vec<(NeuronFacts, usize, FrameNumber, Vec<Row>, Vec<((Sym, Sym, Offset), Conn)>)> = Vec::new();
        for r in read_table(folder, "neurons.csv")? {
            let id: NeuronId = parse(&r[0])?;
            if id as usize != neurons.len() { return Err(bad("neurons.csv: ids in order")); }
            let facts = NeuronFacts { kind: read_kind(&r[1])?, dim: read_opt(&r[2], parse)?, bucket: parse(&r[3])?, channel: read_opt(&r[4], parse)?, axes: read_axes(&r[5])?, alive: read_bool(&r[6])? };
            neurons.push((facts, parse(&r[7])?, parse(&r[8])?, Vec::new(), Vec::new()));
        }
        for r in read_table(folder, "rows.csv")? {
            let id: usize = parse(&r[0])?;
            let nbs = read_list(&r[4], |x| {
                let p: Vec<&str> = x.splitn(4, ':').collect();
                if p.len() != 4 { return Err(bad(x)); }
                Ok(Nb { act: ActRef { frame: parse(p[0])?, index: parse(p[1])? }, sym: parse(p[2])?, off: read_pos(p[3])? })
            })?;
            let row = Row { act: ActRef { frame: parse(&r[1])?, index: parse(&r[2])? }, own: parse(&r[3])?, nbs };
            neurons.get_mut(id).ok_or_else(|| bad("rows.csv: neuron"))?.3.push(row);
        }
        for r in read_table(folder, "connections.csv")? {
            let id: usize = parse(&r[0])?;
            let conn = ((parse(&r[1])?, parse(&r[2])?, read_pos(&r[3])?), Conn { strength: parse(&r[4])?, earnings: parse(&r[5])? });
            neurons.get_mut(id).ok_or_else(|| bad("connections.csv: neuron"))?.4.push(conn);
        }
        self.thalamus.import_neurons(neurons);
        // the dictionary and its indexes
        let mut dict = Dictionary::default();
        dict.next_id = read_key(&s, "next_function")?;
        dict.next_join = read_key(&s, "next_join")?;
        for r in read_table(folder, "poses.csv")? {
            let (n, p) = (parse(&r[0])?, Phase { from: parse(&r[1])?, to: parse(&r[2])?, laps: parse(&r[3])? });
            if read_bool(&r[4])? { dict.sym_index.insert((n, p), SYM_CALL + dict.syms.len() as Sym); }
            dict.syms.push((n, p));
        }
        for r in read_table(folder, "functions.csv")? {
            let f = read_func(&r)?;
            for rel in &f.rels { dict.by_rel.insert(*rel, f.id); }
            dict.funcs.insert(f.id, f);
        }
        dict.note_reaches();
        let mut fids: Vec<FuncId> = dict.funcs.keys().copied().collect();
        fids.sort();
        for id in fids {
            let nodes = dict.funcs[&id].nodes.clone();
            for s in nodes { let n = dict.find_neuron_of(s); let v = dict.by_node.entry(n).or_default(); if !v.contains(&id) { v.push(id); } }
        }
        for r in read_table(folder, "interneurons.csv")? { dict.by_inter.insert(parse(&r[0])?, read_list(&r[1], parse)?); }
        self.thalamus.dict = dict;
        for r in read_table(folder, "retired.csv")? { let f = read_func(&r)?; let until = parse(&r[6])?; self.retired_funcs.insert(f.id, (f, until)); }
        // the ledger, and per interneuron the function that fired it last and its reach
        for r in read_table(folder, "ledger.csv")? {
            let n: NeuronId = parse(&r[0])?;
            self.ledger.push(n);
            if let Some(d) = read_opt(&r[1], parse)? { self.ledger_due.insert(n, d); }
        }
        for r in read_table(folder, "ledger_due.csv")? { self.ledger_due.insert(parse(&r[0])?, parse(&r[1])?); }
        for r in read_table(folder, "last_func.csv")? { self.last_func.insert(parse(&r[0])?, parse(&r[1])?); }
        for r in read_table(folder, "reach_of.csv")? { self.reach_of.insert(parse(&r[0])?, read_pos(&r[1])?); }
        self.last_stats = FrameStats::default();
        Ok(())
    }

    // ── The context ─────────────────────────────────────────────────────────

    fn write_context(&self, folder: &Path) -> Result<(), String> {
        write_table(folder, "context.csv", std::iter::once(vec!["first_frame".to_string(), write_opt(self.first_frame, |f| f.to_string())]))?;
        let frames = &self.thalamus.frames;
        write_table(folder, "frames.csv", frames.iter().map(|f| vec![
            f.no.to_string(),
            write_list(&f.actions_ran, |(c, a)| format!("{}:{}", c, write_act(a))),
            write_list(&f.rewards, |r| format!("{}/{}/{}", r.reward, r.frames, if r.channels.is_empty() { "-".into() } else { r.channels.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(";") })),
        ]))?;
        write_table(folder, "acts.csv", frames.iter().flat_map(|f| f.acts.iter().map(move |a| vec![
            f.no.to_string(), a.neuron.to_string(), a.sym.to_string(), write_kind(a.kind), write_axes(&a.axes), write_pos(&a.pos), write_bool(a.strong),
            write_pos(&a.reach), write_pos(&a.lo), write_pos(&a.hi), write_opt(a.covered_at, |c| c.to_string()), write_list(&a.coverers, write_act),
            write_list(&a.covers, write_act), write_opt(a.fired_by.as_ref(), |fb| fb.func.to_string()), write_opt(a.fired_by.as_ref(), |fb| write_list(&fb.failed, |x| x.to_string())),
            write_list(&a.rels, |br| format!("{}:{}:{}", write_act(&br.other), br.strength, write_pos(&br.off))),
            write_list(&a.inferences, |i| format!("{}:{}:{}:{}:{}", i.target, i.strength, i.estimate, write_bool(i.struck), write_pos(&i.offset))),
            a.opened_at.to_string(), a.level.to_string(), a.height.to_string(), write_bool(a.gone),
        ])))?;
        write_table(folder, "pending_weak.csv", self.pending_weak.iter().map(|(n, p)| vec![n.to_string(), write_pos(p)]))?;
        let sets: [(&str, &FxHashSet<(ActRef, ActRef)>); 4] = [("covered_rel", &self.covered_rel), ("lifted", &self.lifted), ("refused", &self.refused), ("form_pre", &self.form_pre)];
        write_table(folder, "edges.csv", sets.iter().flat_map(|(name, set)| { let mut v: Vec<&(ActRef, ActRef)> = set.iter().collect(); v.sort(); v.into_iter().map(move |e| vec![name.to_string(), write_edge(e)]) }))?;
        let maps: [(&str, &FxHashMap<(ActRef, ActRef), Vec<(ActRef, ActRef)>>); 2] = [("form_src", &self.form_src), ("lift_forms", &self.lift_forms)];
        write_table(folder, "edge_lists.csv", maps.iter().flat_map(|(name, map)| { let mut v: Vec<_> = map.iter().collect(); v.sort_by_key(|x| *x.0); v.into_iter().map(move |(e, l)| vec![name.to_string(), write_edge(e), write_list(l, write_edge)]) }))?;
        let lists: [(&str, &FxHashMap<ActRef, Vec<ActRef>>); 3] = [("cov_inc", &self.cov_inc), ("taken", &self.taken), ("call_thru", &self.call_thru)];
        write_table(folder, "act_lists.csv", lists.iter().flat_map(|(name, map)| { let mut v: Vec<_> = map.iter().collect(); v.sort_by_key(|x| *x.0); v.into_iter().map(move |(a, l)| vec![name.to_string(), write_act(a), write_list(l, write_act)]) }))?;
        let mut v: Vec<_> = self.call_edges.iter().collect();
        v.sort_by_key(|x| *x.0);
        write_table(folder, "act_edges.csv", v.into_iter().map(|(a, l)| vec![write_act(a), write_list(l, write_edge)]))?;
        let mut v: Vec<_> = self.kind_of.iter().collect();
        v.sort_by_key(|x| *x.0);
        write_table(folder, "kinds.csv", v.into_iter().map(|(a, k)| vec![write_act(a), write_bool(*k)]))?;
        let mut v: Vec<_> = self.grown.iter().collect();
        v.sort_by_key(|x| *x.0);
        write_table(folder, "grown.csv", v.into_iter().map(|(a, b)| vec![write_act(a), write_act(b)]))?;
        let mut v: Vec<_> = self.covered_by.iter().collect();
        v.sort_by_key(|x| *x.0);
        write_table(folder, "covered_by.csv", v.into_iter().map(|(a, l)| vec![write_act(a), write_list(l, |(s, o)| format!("{}:{}", s, write_pos(o)))]))
    }

    fn read_context(&mut self, folder: &Path) -> Result<(), String> {
        let s = read_pairs(folder, "context.csv")?;
        self.first_frame = read_opt(s.get("first_frame").ok_or_else(|| bad("first_frame"))?, parse)?;
        let mut frames: std::collections::VecDeque<FrameRec> = std::collections::VecDeque::new();
        for r in read_table(folder, "frames.csv")? {
            let actions_ran = read_list(&r[1], |x| { let (c, a) = x.split_once(':').ok_or_else(|| bad(x))?; Ok((parse(c)?, read_act(a)?)) })?;
            let rewards = read_list(&r[2], |x| {
                let p: Vec<&str> = x.split('/').collect();
                if p.len() != 3 { return Err(bad(x)); }
                Ok(Reward { reward: parse(p[0])?, frames: parse(p[1])?, channels: if p[2] == "-" { Vec::new() } else { p[2].split(';').map(parse).collect::<Result<_, _>>()? } })
            })?;
            frames.push_back(FrameRec { no: parse(&r[0])?, acts: Vec::new(), actions_ran, rewards });
        }
        for r in read_table(folder, "acts.csv")? {
            let no: FrameNumber = parse(&r[0])?;
            let fired_by = match read_opt(&r[13], parse)? { Some(func) => Some(FiredBy { func, failed: read_list(&r[14], parse)? }), None => None };
            let rels = read_list(&r[15], |x| {
                let p: Vec<&str> = x.splitn(4, ':').collect();
                if p.len() != 4 { return Err(bad(x)); }
                Ok(BoardRel { other: ActRef { frame: parse(p[0])?, index: parse(p[1])? }, strength: parse(p[2])?, off: read_pos(p[3])? })
            })?;
            let inferences = read_list(&r[16], |x| {
                let p: Vec<&str> = x.splitn(5, ':').collect();
                if p.len() != 5 { return Err(bad(x)); }
                Ok(Inference { target: parse(p[0])?, strength: parse(p[1])?, estimate: parse(p[2])?, struck: read_bool(p[3])?, offset: read_pos(p[4])? })
            })?;
            let act = Act {
                neuron: parse(&r[1])?, sym: parse(&r[2])?, kind: read_kind(&r[3])?, axes: read_axes(&r[4])?, pos: read_pos(&r[5])?, strong: read_bool(&r[6])?,
                reach: read_pos(&r[7])?, lo: read_pos(&r[8])?, hi: read_pos(&r[9])?, covered_at: read_opt(&r[10], parse)?, coverers: read_list(&r[11], read_act)?,
                covers: read_list(&r[12], read_act)?, fired_by, rels, inferences,
                opened_at: parse(&r[17])?, level: parse(&r[18])?, height: parse(&r[19])?, gone: read_bool(&r[20])?,
            };
            frames.iter_mut().find(|f| f.no == no).ok_or_else(|| bad("acts.csv: frame"))?.acts.push(act);
        }
        self.thalamus.frames = frames;
        self.rebuild_rel_holders();
        self.pending_weak = read_table(folder, "pending_weak.csv")?.iter().map(|r| Ok((parse(&r[0])?, read_pos(&r[1])?))).collect::<Result<_, String>>()?;
        for r in read_table(folder, "edges.csv")? {
            let e = read_edge(&r[1])?;
            match r[0].as_str() { "covered_rel" => &mut self.covered_rel, "lifted" => &mut self.lifted, "refused" => &mut self.refused, "form_pre" => &mut self.form_pre, x => return Err(bad(x)) }.insert(e);
        }
        for r in read_table(folder, "edge_lists.csv")? {
            let (e, l) = (read_edge(&r[1])?, read_list(&r[2], read_edge)?);
            match r[0].as_str() { "form_src" => &mut self.form_src, "lift_forms" => &mut self.lift_forms, x => return Err(bad(x)) }.insert(e, l);
        }
        for r in read_table(folder, "act_lists.csv")? {
            let (a, l) = (read_act(&r[1])?, read_list(&r[2], read_act)?);
            match r[0].as_str() { "cov_inc" => &mut self.cov_inc, "taken" => &mut self.taken, "call_thru" => &mut self.call_thru, x => return Err(bad(x)) }.insert(a, l);
        }
        for r in read_table(folder, "act_edges.csv")? { self.call_edges.insert(read_act(&r[0])?, read_list(&r[1], read_edge)?); }
        for r in read_table(folder, "kinds.csv")? { self.kind_of.insert(read_act(&r[0])?, read_bool(&r[1])?); }
        for r in read_table(folder, "grown.csv")? { self.grown.insert(read_act(&r[0])?, read_act(&r[1])?); }
        for r in read_table(folder, "covered_by.csv")? {
            let l = read_list(&r[1], |x| { let (s, o) = x.split_once(':').ok_or_else(|| bad(x))?; Ok((parse(s)?, read_pos(o)?)) })?;
            self.covered_by.insert(read_act(&r[0])?, l);
        }
        Ok(())
    }
}
