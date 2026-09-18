"""Enumerate distinct physical choices, merging only identical deck copies.

Unlike the upstream rank-only enumeration, keeping one suit versus another
and spending a wildcard instead of a natural card remain different actions.
Single wildcards are level cards. Two wildcards on their own are represented
as a level pair; this is the explicit project rule profile.
"""
from functools import lru_cache
from itertools import product

from .cards import BJ, SJ, SEQV_TO_RANK, is_wildcard, order_of, rank_of, suit_of
from .combos import (BOMB, FULL, PAIR, PASS, PASS_MOVE, PLATE, ROCKET, SFLUSH,
                     SINGLE, STRAIGHT, TRIPLE, TUBE, Move, beats)


def _subsets(cards, size):
    """Choose face-count multisets, using canonical IDs for duplicate faces."""
    groups = {}
    for c in sorted(cards):
        groups.setdefault(c % 54, []).append(c)
    cells = list(groups.values())

    def rec(i, need, selected):
        if need == 0:
            yield tuple(selected)
            return
        if i == len(cells) or sum(map(len, cells[i:])) < need:
            return
        for n in range(min(need, len(cells[i])) + 1):
            yield from rec(i + 1, need - n, selected + cells[i][:n])

    yield from rec(0, size, [])


def enumerate_moves(cards, lv, lead=None):
    wilds = sorted(c for c in cards if is_wildcard(c, lv))
    naturals = [c for c in cards if not is_wildcard(c, lv)]
    by_rank = {r: [c for c in naturals if rank_of(c) == r] for r in range(15)}
    out, seen = [], set()
    following = lead is not None and lead.type != PASS

    @lru_cache(None)
    def choices(rank, count, suit=-1):
        available = by_rank[rank]
        if suit >= 0:
            available = [c for c in available if suit_of(c) == suit]
        result = []
        max_wild = min(count, len(wilds)) if rank < 13 else 0
        for nw in range(max_wild + 1):
            for picked in _subsets(available, count - nw):
                result.append((picked, nw))
        return result

    def pattern(kind, key, groups, suit=-1):
        size = sum(n for r, n in groups)
        # Skip losing patterns before expanding physical card choices.
        probe = Move(kind, key, [0] * size, [])
        if following and not beats(probe, lead, lv):
            return
        opts = [choices(r, n, suit) for r, n in groups]
        if not all(opts):
            return
        for selection in product(*opts):
            nw = sum(x[1] for x in selection)
            if nw > len(wilds):
                continue
            chosen, claim, used = [], [], 0
            for (r, n), (picked, w) in zip(groups, selection):
                chosen.extend(picked)
                chosen.extend(wilds[used:used + w])
                claim.extend([r] * n)
                used += w
            if kind == PAIR and nw == 2 and groups[0][0] != lv:
                continue
            if kind == STRAIGHT:
                # A flush made entirely of natural cards cannot be called a
                # mixed-suit straight. A wildcard may declare another suit.
                fixed_suits = {suit_of(c) for c in chosen if not is_wildcard(c, lv)}
                if nw == 0 and len(fixed_suits) == 1:
                    continue
            identity = (kind, key, tuple(sorted(c % 54 for c in chosen)), suit)
            if identity in seen:
                continue
            seen.add(identity)
            out.append(Move(kind, key, chosen, claim, suit if kind == SFLUSH else None,
                            tuple(c for c in chosen if is_wildcard(c, lv))))

    for face in sorted({c % 54 for c in cards}):
        c = min(c for c in cards if c % 54 == face)
        r = rank_of(c)
        m = Move(SINGLE, order_of(r, lv), [c], [r])
        if not following or beats(m, lead, lv):
            out.append(m)
    for r in range(15):
        pattern(PAIR, order_of(r, lv), [(r, 2)])
        if r < 13:
            pattern(TRIPLE, order_of(r, lv), [(r, 3)])
            for n in range(4, min(10, len(by_rank[r]) + len(wilds)) + 1):
                pattern(BOMB, order_of(r, lv), [(r, n)])
            if not following or lead.type == FULL:
                for p in range(15):
                    if p != r:
                        pattern(FULL, order_of(r, lv), [(r, 3), (p, 2)])
    for kind, length, mult in [(STRAIGHT, 5, 1), (PLATE, 3, 2), (TUBE, 2, 3)]:
        if not following or lead.type == kind:
            for low in range(1, 16 - length):
                pattern(kind, low, [(SEQV_TO_RANK[v], mult) for v in range(low, low + length)])
    for low in range(1, 11):
        for suit in range(4):
            pattern(SFLUSH, low, [(SEQV_TO_RANK[v], 1) for v in range(low, low + 5)], suit)
    pattern(ROCKET, 0, [(SJ, 2), (BJ, 2)])
    out.sort(key=lambda m: (m.type, m.size, m.key, tuple(sorted(m.cards)), m.claim_suit or 0))
    return ([PASS_MOVE] if following else []) + out
