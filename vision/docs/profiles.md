# Profiles

The leaderboard answers *who*. A profile answers everything else about one of them, at
`/profile?name=<name>` — reached by clicking a name on `/stats`.

## A note on empty boards

If a board is empty, check these three before anything else — they look identical on screen and
have nothing to do with each other:

1. **Is the deployed code this code?** `/profile` returning 404 means the box is running an
   older build; the page cannot show data it does not have.
2. **Has counting begun?** Nothing before the epoch in `data/review/stats-epoch.json` is
   counted, ever. The first five minutes after it show nothing at all, deliberately.
3. **Does the window cover anything?** Windows no longer wait for a whole period — they show
   what they have and label it — but a window whose whole span predates the epoch is genuinely
   empty and says so.

It is four things on one page, and they are on one page because each one is a different way
into correcting the same mistake. A wrong figure on a leaderboard is never wrong by itself:
it is wrong because a crossing carries the wrong name, and that crossing came from a frame
somebody can look at. The profile is the path from the figure back to the frame.

## What is stored, and what is not

Only the **bio** is stored (`profiles.py`, a JSON file beside the review data). Everything
else — hours, streaks, the boxes on the chart, the day timelines — is derived from the event
log every time it is asked for, exactly as the leaderboard is, and for the same reason: a name
corrected today changes what last Tuesday should say. A stored total would be wrong from the
moment of the correction until somebody remembered to rebuild it.

The bio is stored because it could not be otherwise. No doorway will ever work out that
somebody sits by the window and is learning Finnish.

## The name

The heading is the label the system uses, and editing it goes through the same
`POST /api/rename` the labelling page uses: the gallery, every frame filed under the old
spelling, every crossing in the event log, whoever is inside right now — and the bio.

The bio moving matters more than it sounds. Spelling somebody's name properly should not cost
them their profile. When *both* names have a bio — which is what a merge of two real entries
looks like — the two are joined rather than one dropped. Nothing a person wrote about
themselves is discarded because of a click somewhere else.

**A rename now also rebuilds the matching set** (`ReviewQueue.rename` calls
`refresh_matching`). It did not, and that was a real fault rather than an omission: the
matching set is keyed by name, so the references moved to the corrected spelling while
recognition went on matching — and announcing — the old one until a restart or the next label
happened to rebuild it. The correction appeared everywhere a human looked and nowhere the
system did.

## The picture

The face **closest to that person's own average**, preferring the ones recognition is
actually using (`web/faces.py: representative`).

* Closest to the average rather than newest, because the average *is* what the system thinks
  this person looks like, so the nearest face to it is the most ordinary picture of them there
  is. The newest is whatever the door caught this morning, as likely to be a blurred shoulder
  as a portrait.
* In-use first, even when an unused face sits marginally closer: a face is unused because it
  is too old or was ruled out by hand, and a profile should not be represented by a picture the
  system has stopped believing in.
* No picture when there is none. A person can be entirely real and have reference vectors with
  no crop saved; inventing a portrait for them is not on offer.

Clicking it opens `/gallery?name=<name>` — every frame filed under this person, which of them
recognition is matching against, and why each one is or is not. That page can now also
**correct who a frame is**, not just whether it is used: two different questions, so two
different controls.

## The chart

A box a day for the last 53 weeks, copying GitHub's contribution graph — worth copying because
it answers "is this a habit" at a glance, and because everybody who will look at this page has
already learnt to read it. `activity.py` derives it; `static/calendar.js` and `calendar.css`
draw it, as a standalone widget so a second page needing one does not need a copy.

Three things about it are choices rather than arithmetic:

* **A day runs 04:30 to 04:30**, as everywhere else here. Somebody who left at one in the
  morning gets one dark box for a long day, not two pale ones for two half days. A visit that
  crosses the boundary is split across the days it touches.
* **The shades are relative to that person's own busiest day** in the chart. That is what makes
  the pattern readable both for somebody who comes in for an hour and for somebody who lives
  here — and it means a shade cannot be compared between two people's charts, only within one.
  Any time at all reaches the first shade, so a ten-minute visit is a visible box rather than
  rounding away into looking like a day off.
* **Days before counting began are drawn as empty dashed boxes and say so.** An empty box then
  means "nobody was looking", which is not the same as "they were not here", and a chart that
  renders the two identically is quietly lying about the emptiest part of itself.

## One day

Clicking a box opens that day: every entry and exit in order, each with the frame it was
decided from, what that frame is filed as *now*, and a way to say it is wrong.

Two details earn their place:

* A crossing whose frame is filed under a **different** name than the profile is flagged. That
  disagreement is the thing this page exists to surface — it usually means a frame was
  relabelled and the crossing was left behind, or the reverse.
* A crossing with **no frame** says so rather than offering a dead control. Those were counted
  from the doorframe alone, and there is nothing to look at or relabel.

Correcting a label here goes through `POST /api/label`, the same route the labelling page uses,
so it moves the face between galleries *and* rewrites the crossing — the hours follow the
correction. The page then reloads both the day and the chart, because a correction changes the
shade of the box as well as the name on the row.

## Where it lives

| Concern | File |
| --- | --- |
| Per-day activity and one day's crossings | `activity.py` |
| The bio store, and carrying it through a rename | `profiles.py` |
| Totals, visits and the epoch, shared with the leaderboard | `presence.py` |
| One person's gallery, and which face represents them | `web/faces.py` |
| The page and its four endpoints | `web/profile.py` |
| Cache-busting token, shared by both routing modules | `web/assets.py` |
| Markup, styles, behaviour | `templates/profile.html`, `static/profile.{css,js}`, `static/calendar.{css,js}` |

## Endpoints

```
GET  /profile?name=<name>                 the page
GET  /api/profile/<name>                  name, bio, picture, totals, streak, inside
POST /api/bio                             {name, bio} -- empty clears it, over 500 chars is refused
GET  /api/profile/<name>/activity?weeks=  a box per day
GET  /api/profile/<name>/day/<YYYY-MM-DD> that day's entries and exits, with their frames
```

The leaderboard gained `days_covered`, `days_in_window` and `partial` at the same time, for the
same reason as the chart's dashed boxes: the shape of what is missing is part of the answer.

The dated endpoints work without an event store — a labelling-only deployment has no crossings,
and a profile with no chart is a smaller loss than no profile.
