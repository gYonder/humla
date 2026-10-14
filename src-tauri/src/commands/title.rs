//! Automatic Note titles (#90).
//!
//! A Note whose Title Humla owns — empty, a `Recording 19 Aug 14:32`
//! timestamp, or the `Imported audio` import fallback — gets a short phrase
//! derived from its own content once that content settles. The
//! model call is deliberately small and cheap: the head of the
//! notes and a head/middle/tail sample of the transcript, four to six words back,
//! no streaming.
//!
//! Three things this is *not*, all of them settled during the issue's
//! grilling:
//!
//! - **Not driven by the summary.** Summarising in Humla is a button the user
//!   presses, so a summary-completion trigger would leave the motivating case
//!   — a menu-bar recorder who never opens the note — on its timestamp
//!   forever.
//! - **Not live.** The title needs the closing exchanges, so firing during
//!   the recording would title from an incomplete tail as well as contend
//!   with local Whisper for GPU — after stop is the cheap moment.
//! - **Not loud.** Failure is silent. A user who never asked for a Title must
//!   not be told one failed, and a local-only user with no LLM configured
//!   would otherwise be toasted after every recording.

use super::summary::{language_directive, resolve_auto, resolve_provider, summary_provider_id};
use super::{DEFAULT_LANGUAGE, TITLE_FALLBACK_MODEL};
use crate::db::{self, NotePatch};
use crate::menubar::is_replaceable_title;
use crate::openai;
use crate::recording::TitleStatusPayload;
use crate::AppState;
use tauri::{AppHandle, Emitter, Manager, State};

/// Hard cap on a generated Title. The prompt asks for four to six words; this
/// is the backstop for a model that answers with a sentence.
const TITLE_MAX_CHARS: usize = 60;

/// Past this, the model didn't answer the question — it wrote prose. Truncating
/// that to 60 characters would produce a mangled fragment, which is a worse
/// Title than the timestamp it would replace, so we reject instead.
const TITLE_JUNK_CHARS: usize = 2 * TITLE_MAX_CHARS;

/// How much of each content block the model sees. Notes contribute their head
/// — the agenda the user typed. The transcript contributes a head+tail sample
/// within the same budget (see below): sending the whole transcript would
/// cost real tokens, but head-only always saw the smalltalk warm-up.
const BLOCK_CHAR_BUDGET: usize = 750;

/// Transcript opener kept verbatim. Enough to catch an agenda stated up front,
/// small enough that weather-and-weekend smalltalk can't fill the budget.
/// Borrowed from meetily's "analyze the entire transcript" insight: the title
/// must see past the opening exchange, so most of the budget goes to the
/// middle and tail where the substance lands.
const TRANSCRIPT_HEAD_CHARS: usize = 150;

/// Centered transcript window. Covers the discussion neither end sees — the
/// important-decision-in-the-middle case a head+tail sample misses entirely.
const TRANSCRIPT_MID_CHARS: usize = 300;

/// Transcript tail kept verbatim. Together with the head, the middle and the
/// two `…` joiners this stays under `BLOCK_CHAR_BUDGET`, so the fix costs no
/// extra tokens.
const TRANSCRIPT_TAIL_CHARS: usize = 250;

/// Turn raw model output into a Title, or `None` when there is nothing usable
/// in it.
///
/// `None` is the "leave the existing Title alone" signal, and it is the type's
/// default rather than a convention the caller has to remember: there is no
/// worse-Title fallback anywhere in this path.
pub(crate) fn clean_title(raw: &str) -> Option<String> {
    // Single line: newlines collapse to spaces, then whitespace runs collapse
    // to one. A small model asked for six words will still sometimes answer
    // with a heading and a blank line.
    let mut t: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    // Peel until nothing more comes off. The three artifacts nest in either
    // order — `"Title: Kickoff"` and `"Kickoff".` both occur — so a fixed
    // sequence would leave whichever one it happened to check first.
    loop {
        let before = t.clone();
        t = strip_label_prefix(&t).to_string();
        t = strip_wrapping_quotes(&t).to_string();
        // Asked for no final period, and told anyway. Only the trailing full
        // stop — a title genuinely ending in `?` or `!` keeps it.
        t = t.trim_end_matches(['.', '\u{2026}']).trim().to_string();
        if t == before {
            break;
        }
    }
    let t = t.as_str();
    if t.is_empty() || !t.chars().any(char::is_alphanumeric) {
        return None;
    }
    if t.chars().count() > TITLE_JUNK_CHARS {
        return None;
    }
    Some(truncate_chars(t, TITLE_MAX_CHARS))
}

/// Peel a `Title:` / `Tittel:` label a model prepended despite being asked not
/// to. Only when something follows it — a title that IS the bare word is junk,
/// and `clean_title`'s emptiness check should be the one to say so.
fn strip_label_prefix(t: &str) -> &str {
    for label in ["title:", "tittel:"] {
        // Compare by chars, not by byte prefix — `t` may well start with a
        // multibyte character, and `&t[..label.len()]` would panic on it.
        if t.chars().count() > label.len()
            && t.chars()
                .zip(label.chars())
                .all(|(a, b)| a.to_ascii_lowercase() == b)
        {
            return t[label.len()..].trim_start();
        }
    }
    t
}

/// Peel one matching pair of wrapping quotes, straight or typographic.
fn strip_wrapping_quotes(t: &str) -> &str {
    const PAIRS: [(char, char); 5] =
        [('"', '"'), ('\'', '\''), ('\u{201c}', '\u{201d}'), ('\u{00ab}', '\u{00bb}'), ('\u{2018}', '\u{2019}')];
    let mut chars = t.chars();
    let (Some(first), Some(last)) = (chars.next(), chars.next_back()) else { return t };
    for (open, close) in PAIRS {
        if first == open && last == close {
            return t[open.len_utf8()..t.len() - close.len_utf8()].trim();
        }
    }
    t
}

/// Cut to at most `max` chars, on a `char` boundary and, where one is close
/// enough, on a word boundary — so a cap that lands mid-word doesn't leave a
/// half word behind.
fn truncate_chars(t: &str, max: usize) -> String {
    if t.chars().count() <= max {
        return t.to_string();
    }
    let cut = t.char_indices().nth(max).map(|(i, _)| i).unwrap_or(t.len());
    let head = &t[..cut];
    match head.rfind(' ') {
        // Only when the word boundary keeps most of the budget — otherwise a
        // long final word would throw away half the title.
        Some(i) if i >= max / 2 => head[..i].trim_end().to_string(),
        _ => head.trim_end().to_string(),
    }
}

/// The two labelled blocks the model reads: head of the notes, head/middle/tail
/// of the transcript.
///
/// Reuses the summary path's `[Notater]` / `[Transkripsjon]` convention
/// including its `(ingen)` placeholder: both blocks are always sent, so a
/// typed-only, a recorded-only and a mixed Note all take one path with no
/// precedence rule — and a thinking model isn't left wondering whether a
/// missing block is absent or hidden.
///
/// The transcript is the fix for smalltalk-dominated titles: `head_chars`
/// alone always saw `[0, 750)` — precisely the weather/how-are-you warm-up.
/// Sampling head + middle + tail within the same budget lets the model see the
/// substance no single end holds, mirroring what meetily gets for free by
/// titling from its summary rather than the raw stream.
pub(crate) fn title_input(body_text: &str, transcript: &str) -> String {
    let notes = head_chars(body_text, BLOCK_CHAR_BUDGET);
    let spoken = sample_transcript(transcript);
    let notes_block = if notes.is_empty() { "(ingen)".to_string() } else { notes };
    let transcript_block = if spoken.is_empty() { "(ingen)".to_string() } else { spoken };
    format!("[Notater]\n{notes_block}\n\n[Transkripsjon]\n{transcript_block}")
}

/// Head + middle + tail sample of the transcript within `BLOCK_CHAR_BUDGET`.
///
/// Short transcripts pass through untouched. Long ones keep the first
/// `TRANSCRIPT_HEAD_CHARS` (agenda cue), a centered `TRANSCRIPT_MID_CHARS`
/// window (the discussion neither end sees) and the last
/// `TRANSCRIPT_TAIL_CHARS` (decisions and wrap-up), joined with ellipses so
/// the model knows speech was cut. All cuts land on word and char boundaries.
/// Near the budget boundary the windows can overlap by a few words — harmless
/// duplication in a title prompt.
fn sample_transcript(text: &str) -> String {
    let t = text.trim();
    if t.chars().count() <= BLOCK_CHAR_BUDGET {
        return t.to_string();
    }
    let head = cut_head_word_boundary(t, TRANSCRIPT_HEAD_CHARS);
    let middle = cut_middle_word_boundary(t, TRANSCRIPT_MID_CHARS);
    let tail = cut_tail_word_boundary(t, TRANSCRIPT_TAIL_CHARS);
    format!("{head}\n…\n{middle}\n…\n{tail}")
}

/// First `budget` chars of `text`, trimmed, cut on a `char` boundary.
fn head_chars(text: &str, budget: usize) -> String {
    let t = text.trim();
    let cut = t.char_indices().nth(budget).map(|(i, _)| i).unwrap_or(t.len());
    t[..cut].trim_end().to_string()
}

/// Head cut that steps back to the previous whitespace, so the fragment never
/// ends mid-word and pulls the model toward a dangling fragment.
fn cut_head_word_boundary(text: &str, budget: usize) -> String {
    let raw = head_chars(text, budget);
    match raw.rfind(char::is_whitespace) {
        Some(i) if i >= budget / 2 => raw[..i].trim_end().to_string(),
        _ => raw,
    }
}

/// Byte index of the first whole word at or after `start_byte` (`start_byte`
/// must be a char boundary): unchanged when already on a boundary — start of
/// text, or preceded by whitespace — otherwise stepped forward to the next
/// whitespace, unless that step would eat most of `budget` (one very long
/// word). Whitespace-aware (`is_whitespace`, not just `' '`) so newlines and
/// tabs count as boundaries too.
fn word_start_at_or_after(t: &str, start_byte: usize, budget: usize) -> usize {
    if start_byte == 0
        || t[..start_byte]
            .chars()
            .next_back()
            .is_some_and(char::is_whitespace)
    {
        return start_byte;
    }
    match t[start_byte..]
        .char_indices()
        .find(|(_, c)| c.is_whitespace())
    {
        Some((i, c)) if i < budget / 2 => start_byte + i + c.len_utf8(),
        _ => start_byte,
    }
}

/// Centered `budget`-char window of `text`, on word boundaries where cheap.
/// Shares the tail's boundary rule: a window already starting on a boundary
/// keeps its first word whole.
fn cut_middle_word_boundary(text: &str, budget: usize) -> String {
    let t = text.trim();
    let total = t.chars().count();
    if total <= budget {
        return t.to_string();
    }
    let skip = (total - budget) / 2;
    let start_byte = t.char_indices().nth(skip).map(|(i, _)| i).unwrap_or(0);
    let from = word_start_at_or_after(t, start_byte, budget);
    cut_head_word_boundary(&t[from..], budget)
}

/// Last `budget` chars of `text`, starting on the next word boundary so the
/// fragment never starts mid-word — unless it already does, in which case the
/// first word is kept whole. Char-boundary safe for multibyte text.
fn cut_tail_word_boundary(text: &str, budget: usize) -> String {
    let t = text.trim();
    let total = t.chars().count();
    if total <= budget {
        return t.to_string();
    }
    let skip = total - budget;
    let start_byte = t.char_indices().nth(skip).map(|(i, _)| i).unwrap_or(0);
    let from = word_start_at_or_after(t, start_byte, budget);
    t[from..].trim_start().to_string()
}

/// The system prompt. Minimal on purpose — a constraint-heavy prompt becomes a
/// checklist a small thinking model re-litigates instead of following. The two
/// added sentences are meetily's whole trick spelled out for a three-part
/// sample: ignore the pleasantries around the cuts, read the parts as one
/// excerpt — and when smalltalk is all there is, name that instead of
/// inventing a topic.
fn title_prompt(language: &str) -> String {
    format!(
        "Give the note below a short title naming what it is about. \
         Four to six words. Answer with the title alone — no quotes, no label, \
         no final period, no explanation. Name the main topic, decision or \
         purpose — not the greetings, weather or check-ins at the start, nor \
         the sign-offs at the end; each … marks cut-out speech, so read the \
         parts as one excerpt. If smalltalk is all the note holds, name that \
         rather than inventing a topic.\n\n{}",
        language_directive(language)
    )
}

/// Title `note_id` if its Title is still one of Humla's — the post-stop
/// recording chain's entry point, spawned rather than awaited.
///
/// Swallows every failure, which is the whole point of it being separate from
/// [`note_generate_title`]: nobody asked for this Title, so nobody is told when
/// one doesn't arrive. See the module docs.
pub(crate) async fn generate_note_title(app: AppHandle, note_id: String) {
    if let Err(e) = try_generate_note_title(&app, &note_id, false).await {
        eprintln!("[title] note {note_id}: {e:#}");
    }
}

async fn try_generate_note_title(
    app: &AppHandle,
    note_id: &str,
    force: bool,
) -> anyhow::Result<Option<String>> {
    let state: State<AppState> = app.state();
    // Which provider first, then that provider's key — the summary path's rule,
    // so a title never opens a Keychain slot its provider doesn't use.
    let provider_id = {
        let conn = state.db.lock();
        let n = db::get_note(&conn, note_id)?;
        if !force && !is_replaceable_title(&n.title) {
            return Ok(None);
        }
        summary_provider_id(&conn, &n)?
    };
    let api_key = match provider_id.keychain_account() {
        Some(_) => super::read_provider_api_key(&state, provider_id.as_str())
            .map_err(|e| anyhow::anyhow!("{e}"))?,
        None => None,
    };
    let (mut provider, language, note) = {
        let conn = state.db.lock();
        let n = db::get_note(&conn, note_id)?;
        if !force && !is_replaceable_title(&n.title) {
            return Ok(None);
        }
        let p = resolve_provider(&conn, &n, api_key)?;
        let global = db::get_setting(&conn, "language")?
            .unwrap_or_else(|| DEFAULT_LANGUAGE.to_string());
        // Same fallback rule as the summary: the note's language wins, empty
        // means "follow the global default".
        let lang = if n.language.trim().is_empty() { global } else { n.language.clone() };
        (p, resolve_auto(&lang, &n), n)
    };
    if crate::html_text::html_to_text(&note.body).trim().is_empty()
        && note.transcript.trim().is_empty()
    {
        return Ok(None);
    }
    // Thinking on a five-word title is waste, whatever the user set for
    // summaries.
    provider.think = false;
    if provider.base_url == openai::BASE && openai::is_reasoning_model(&provider.model) {
        provider.model = TITLE_FALLBACK_MODEL.to_string();
    }
    // Everything cheap and every refusal is behind us: from here a model call
    // is definitely happening, which is exactly what the Note view shimmers
    // over. Bracketing any earlier would flash a placeholder on notes we then
    // decline to title.
    emit_title_status(app, note_id, true);
    let written = write_generated_title(app, note_id, &provider, &language, &note, force).await;
    emit_title_status(app, note_id, false);
    written
}

/// Ask the model, clean the answer, and write it. Split out so
/// [`try_generate_note_title`] can bracket exactly this with `title_status` —
/// the `?`s in here are why: an early return must still clear the shimmer.
async fn write_generated_title(
    app: &AppHandle,
    note_id: &str,
    provider: &super::summary::ResolvedProvider,
    language: &str,
    note: &db::Note,
    force: bool,
) -> anyhow::Result<Option<String>> {
    let state: State<AppState> = app.state();
    let body_text = crate::html_text::html_to_text(&note.body);
    // No streaming into the summary panel: this call is not a summary and must
    // not touch its UI.
    let raw = provider
        .complete(&title_prompt(language), &title_input(&body_text, &note.transcript), |_| {})
        .await?;
    let Some(title) = clean_title(&raw) else {
        eprintln!("[title] note {note_id}: model returned nothing usable, keeping the title");
        return Ok(None);
    };
    {
        let conn = state.db.lock();
        // Re-check under the lock: the user may have typed a title of their own
        // while the model was working, and that one wins.
        let current = db::get_note(&conn, note_id)?;
        if !force && !is_replaceable_title(&current.title) {
            return Ok(None);
        }
        db::update_note(&conn, note_id, &NotePatch {
            title: Some(title.clone()),
            ..Default::default()
        })?;
    } // drop the db guard before pinging sync (see the SyncObserver contract)
    // A Title is real content a teammate must see, so unlike the derived
    // local-only writes (`set_detected_language`, `speakers`) this one bumps
    // `updated_at` and goes to the workspace.
    state.sync.note_upserted(note_id);
    // Backend write — the notes store never asked for it, so tell it. This is
    // what puts the new Title in the sidebar, and (via the Note view's guarded
    // adopt) in an open note's title box.
    let _ = app.emit("notes_changed", ());
    eprintln!("[title] note {note_id}: titled {title:?}");
    Ok(Some(title))
}

fn emit_title_status(app: &AppHandle, note_id: &str, active: bool) {
    let _ = app.emit("title_status", TitleStatusPayload {
        note_id: note_id.to_string(),
        active,
    });
}

/// Title a Note from the frontend. Two callers, and `force` is the whole of the
/// difference between them:
///
/// - `force: true` — the ⋯ menu's **Regenerate title**. The escape hatch for a
///   bad automatic Title, and the only way to re-title a Note recorded in
///   several takes (the automatic path is one-shot by construction: once it
///   writes a Title, that Title is no longer replaceable). Overwrites whatever
///   is there, including a Title the user typed, because the user asked.
/// - `force: false` — the Note view's body-settled checkpoint, which is what
///   titles a Note that was typed and never recorded. Eligibility decides, and
///   an ineligible Note costs nothing: this returns before any model call.
///
/// Returns the new Title so the open Note view can put it in its draft directly
/// — the guarded adopt that carries the automatic path deliberately refuses to
/// overwrite a user-owned title, and regenerate asks for exactly that.
///
/// Reports failure, unlike the post-stop path, because a `force` call is a
/// button press and is owed an answer.
#[tauri::command]
pub async fn note_generate_title(
    app: AppHandle,
    note_id: String,
    force: bool,
) -> Result<Option<String>, String> {
    try_generate_note_title(&app, &note_id, force)
        .await
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_a_well_formed_title_verbatim() {
        assert_eq!(clean_title("Kickoff with Hege").as_deref(), Some("Kickoff with Hege"));
    }

    #[test]
    fn collapses_a_multi_line_answer_onto_one_line() {
        assert_eq!(
            clean_title("\n  Kickoff\n  with   Hege \n").as_deref(),
            Some("Kickoff with Hege")
        );
    }

    #[test]
    fn strips_wrapping_quotes_and_a_trailing_period() {
        assert_eq!(clean_title("\"Kickoff with Hege\".").as_deref(), Some("Kickoff with Hege"));
        assert_eq!(clean_title("“Kickoff with Hege”").as_deref(), Some("Kickoff with Hege"));
        assert_eq!(clean_title("«Oppstart med Hege»").as_deref(), Some("Oppstart med Hege"));
        assert_eq!(clean_title("'Kickoff with Hege'").as_deref(), Some("Kickoff with Hege"));
    }

    #[test]
    fn keeps_question_and_exclamation_marks() {
        // Only the full stop is an artifact of the model ignoring the prompt.
        assert_eq!(clean_title("Should we ship on Friday?").as_deref(), Some("Should we ship on Friday?"));
    }

    #[test]
    fn strips_a_label_the_model_added() {
        assert_eq!(clean_title("Title: Kickoff with Hege").as_deref(), Some("Kickoff with Hege"));
        assert_eq!(clean_title("\"Tittel: Oppstart med Hege\"").as_deref(), Some("Oppstart med Hege"));
    }

    #[test]
    fn junk_leaves_the_title_alone() {
        // Every one of these must reach the caller's "change nothing" path.
        assert_eq!(clean_title(""), None);
        assert_eq!(clean_title("   \n\t "), None);
        assert_eq!(clean_title("\"\""), None);
        assert_eq!(clean_title("..."), None);
        assert_eq!(clean_title("---"), None);
    }

    #[test]
    fn a_paragraph_leaves_the_title_alone() {
        // A model that answered with prose did not answer the question, and a
        // 60-char slice of prose is a worse title than the timestamp it would
        // replace.
        let paragraph = "This note covers the kickoff meeting with Hege, where the team \
                         discussed the launch timeline, the remaining blockers, and who \
                         would own each of them.";
        assert_eq!(clean_title(paragraph), None);
    }

    #[test]
    fn a_slightly_long_title_is_capped_not_rejected() {
        // Between "a title" and "a paragraph": trim it rather than lose it.
        let long = "Kickoff meeting with Hege about the launch timeline and blockers";
        let t = clean_title(long).unwrap();
        assert!(t.chars().count() <= TITLE_MAX_CHARS, "{t:?}");
        assert!(long.starts_with(&t), "{t:?}");
    }

    #[test]
    fn a_cap_never_splits_a_multibyte_character() {
        // 61 two-byte chars: every cut inside this string is mid-character
        // unless the cut is made on char indices.
        let wide = "æ".repeat(61);
        let t = clean_title(&wide).unwrap();
        assert_eq!(t.chars().count(), TITLE_MAX_CHARS);
        // Reaching this line at all means the slice was valid UTF-8.
        assert!(t.chars().all(|c| c == 'æ'));
    }

    #[test]
    fn a_multibyte_title_is_capped_on_a_word_boundary() {
        let long = "Oppstartsmøte med Hege om lanseringsplanen og gjenstående blokkere";
        let t = clean_title(long).unwrap();
        assert!(t.chars().count() <= TITLE_MAX_CHARS, "{t:?}");
        assert!(!t.ends_with(' '), "{t:?}");
        assert!(long.starts_with(&t), "{t:?}");
    }

    #[test]
    fn both_blocks_are_always_sent() {
        // A typed-only note and a recorded-only note take the same path — the
        // empty side says "(ingen)" rather than going missing.
        let typed = title_input("agenda: launch", "");
        assert!(typed.contains("[Notater]\nagenda: launch"), "{typed}");
        assert!(typed.contains("[Transkripsjon]\n(ingen)"), "{typed}");

        let recorded = title_input("", "Hege: hei");
        assert!(recorded.contains("[Notater]\n(ingen)"), "{recorded}");
        assert!(recorded.contains("[Transkripsjon]\nHege: hei"), "{recorded}");
    }

    #[test]
    fn each_block_is_capped_at_its_own_budget() {
        let long = "a ".repeat(2000);
        let input = title_input(&long, &long);
        // Two capped blocks plus the labels — nowhere near the raw input.
        assert!(input.len() < 4 * BLOCK_CHAR_BUDGET, "{}", input.len());
        assert!(input.contains("[Notater]"));
        assert!(input.contains("[Transkripsjon]"));
    }

    #[test]
    fn a_cap_on_a_block_never_splits_a_multibyte_character() {
        let wide = "æ".repeat(BLOCK_CHAR_BUDGET + 40);
        let input = title_input(&wide, "");
        // Would have panicked on a byte-index slice.
        assert!(input.contains(&"æ".repeat(BLOCK_CHAR_BUDGET)));
    }

    #[test]
    fn the_prompt_carries_the_notes_language() {
        assert!(title_prompt("no").contains("norsk"));
        assert!(title_prompt("en").contains("English"));
        // On `auto` the only cue is the transcript block — same rule as #167.
        assert!(title_prompt("auto").contains("transcript"));
    }

    #[test]
    fn sampling_budget_never_exceeds_block_budget() {
        // Locks the "costs no extra tokens" claim: head + joiner + middle +
        // joiner + tail must fit inside one block, so a constants bump can't
        // silently rot it.
        assert!(
            TRANSCRIPT_HEAD_CHARS + 3 + TRANSCRIPT_MID_CHARS + 3 + TRANSCRIPT_TAIL_CHARS
                <= BLOCK_CHAR_BUDGET
        );
    }

    #[test]
    fn the_prompt_ignores_pleasantries_at_both_ends() {
        // Meetily's trick spelled out for a three-part sample: the faithful
        // answer to "what is this about" must not be the weather opener or
        // the goodbye — unless smalltalk is all there is.
        let p = title_prompt("en");
        assert!(p.contains("nor the sign-offs at the end"), "{p}");
        assert!(p.contains("read the parts as one excerpt"), "{p}");
        assert!(p.contains("If smalltalk is all the note holds"), "{p}");
    }

    #[test]
    fn short_transcript_passes_through_untouched() {
        let short = "You: hei Hege. Hege: hei, hvordan går det?";
        assert_eq!(sample_transcript(short), short);
        assert_eq!(sample_transcript(""), "");
    }

    #[test]
    fn long_transcript_keeps_head_and_tail_within_budget() {
        // Opener smalltalk + substance at the end: the exact failure mode from
        // the issue (title about weather/feelings instead of the meeting).
        let opener =
            "You: hei, fint vær i dag? Hege: ja, nydelig sol! Hvordan har du det? ".repeat(20);
        let substance =
            "Beslutning: lanseringsplanen utsettes til fredag, Hege eier blokkerne. ".repeat(20);
        let transcript = format!("{opener} {substance}");
        assert!(transcript.chars().count() > BLOCK_CHAR_BUDGET);
        let sampled = sample_transcript(&transcript);
        // Head cue preserved, tail substance visible, middle cut marked.
        assert!(sampled.contains("fint vær"), "{sampled}");
        assert!(sampled.contains("lanseringsplanen"), "{sampled}");
        assert!(sampled.contains("…"), "{sampled}");
        assert!(
            sampled.chars().count() <= BLOCK_CHAR_BUDGET,
            "budget blown: {}",
            sampled.chars().count()
        );
    }

    #[test]
    fn transcript_just_over_budget_keeps_both_ends() {
        // Near-boundary: head and tail come closest here, so this is where an
        // overlap/duplication bug would show.
        let transcript = format!(
            "{} {}",
            "værprat og helgeplaner. ".repeat(25),
            "lanseringsplanen utsettes til fredag. ".repeat(8)
        );
        assert!(transcript.chars().count() > BLOCK_CHAR_BUDGET);
        let sampled = sample_transcript(&transcript);
        assert!(sampled.contains("værprat"), "{sampled}");
        assert!(sampled.contains("lanseringsplanen"), "{sampled}");
        assert!(
            sampled.chars().count() <= BLOCK_CHAR_BUDGET,
            "{}",
            sampled.chars().count()
        );
    }

    #[test]
    fn transcript_sample_never_starts_or_ends_mid_word() {
        let transcript = ("supercalifragilisticexpialidocious ".repeat(30))
            + &"lanseringsplanen eies av Hege ".repeat(30);
        let sampled = sample_transcript(&transcript);
        // The joiner splits head from tail; neither side dangles half a word.
        for part in sampled.split("\n…\n") {
            assert!(!part.starts_with("uper"), "{part}");
            assert!(!part.ends_with("supercalifragilisti"), "{part}");
        }
    }

    #[test]
    fn transcript_tail_is_multibyte_safe() {
        let wide = "æ ".repeat(2000);
        let sampled = sample_transcript(&wide);
        // Would have panicked on a byte-index slice; reaching here is the test.
        assert!(sampled.contains('æ'));
        assert!(sampled.chars().count() <= BLOCK_CHAR_BUDGET);
    }

    #[test]
    fn tail_keeps_a_word_it_starts_on() {
        // Regression: the tail must not delete a complete word when the cut
        // already lands on a word boundary. Alignment locked: skip lands
        // exactly on the 'O' of ORION, preceded by a space.
        let text = format!(
            "{}ORION vedtaket {}",
            "p ".repeat(100),
            "q ".repeat(117) + "q"
        );
        assert_eq!(text.chars().count(), 450);
        let tail = cut_tail_word_boundary(&text, 250);
        assert!(tail.contains("ORION"), "{tail}");
    }

    #[test]
    fn tail_boundary_counts_newlines_and_tabs() {
        // Same boundary rule, but the preceding whitespace is a newline rather
        // than a space — `is_whitespace`, not just `' '`.
        let text = format!("{}a\nORION x {}", "a ".repeat(99), "b ".repeat(121));
        assert_eq!(text.chars().count(), 450);
        let tail = cut_tail_word_boundary(&text, 250);
        assert!(tail.contains("ORION"), "{tail}");
    }

    #[test]
    fn head_cut_steps_back_over_newlines() {
        // The last whitespace before the cut is a newline: stepping back to
        // an earlier space would throw away a whole valid word.
        assert_eq!(cut_head_word_boundary("aaa bbb\nccc ddd", 11), "aaa bbb");
    }

    #[test]
    fn long_meeting_keeps_opener_middle_and_closer() {
        // The important-decision-in-the-middle case: head+tail alone would
        // show smalltalk and farewells and miss the decision entirely.
        let opener = "fint vær i dag, hvordan står det til? ".repeat(40);
        let middle = "ORION-vedtaket: utsettelse til fredag, Hege eier blokkerne. ".repeat(15);
        let closer = "takk for i dag, ha det bra, god helg! ".repeat(40);
        let transcript = format!("{opener}{middle}{closer}");
        assert!(transcript.chars().count() > BLOCK_CHAR_BUDGET);
        let sampled = sample_transcript(&transcript);
        assert!(sampled.contains("fint vær"), "{sampled}");
        assert!(sampled.contains("ORION-vedtaket"), "{sampled}");
        assert!(sampled.contains("god helg"), "{sampled}");
        assert_eq!(sampled.matches('…').count(), 2, "{sampled}");
        assert!(
            sampled.chars().count() <= BLOCK_CHAR_BUDGET,
            "{}",
            sampled.chars().count()
        );
    }
}
