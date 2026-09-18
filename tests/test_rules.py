import random
from collections import Counter

import pytest

from fabledan.cards import BJ, SJ, is_wildcard, order_of, rank_of
from fabledan.combos import (
    BOMB,
    FULL,
    PAIR,
    PASS,
    PLATE,
    ROCKET,
    SFLUSH,
    SINGLE,
    STRAIGHT,
    TRIPLE,
    TUBE,
    Move,
    beats,
    claim_ids,
    classify_claim,
    gen_moves,
)
from fabledan.engine import GuandanRound, default_return_card, forced_tribute_card, play_round
from guandan.agents import TeamRuleAgent


def card(rank, suit=0, deck=0):
    return rank * 4 + suit + deck * 54


def test_all_cards_and_level_order():
    assert Counter(rank_of(c) for c in range(108)) == {**{r: 8 for r in range(13)}, SJ: 2, BJ: 2}
    for level in range(13):
        assert sum(is_wildcard(c, level) for c in range(108)) == 2
        assert order_of(BJ, level) > order_of(SJ, level) > order_of(level, level)
        assert all(order_of(level, level) > order_of(r, level) for r in range(13) if r != level)


def test_suits_and_optional_wildcard_spending_are_not_pruned():
    # Keep either suit for a future flush, and retain the option to spend wild.
    hand = [card(5, 1), card(5, 2), card(5, 3), card(1)]
    moves = gen_moves(hand, 1)
    pairs = [m for m in moves if m.type == PAIR and m.claim_ranks == [5, 5]]
    assert len(pairs) == 6
    assert any(card(1) in m.cards for m in pairs)
    assert any(card(1) not in m.cards for m in pairs)


def test_natural_level_triple_and_two_wild_pair():
    moves = gen_moves([card(1), card(1, 1), card(1, 2)], 1)
    assert any(m.type == TRIPLE and len(m.cards) == 3 for m in moves)
    moves = gen_moves([card(1), card(1, 0, 1)], 1)
    assert [m.claim_ranks for m in moves if m.type == PAIR] == [[1, 1]]


def test_wildcards_cannot_impersonate_jokers():
    for m in gen_moves([53, 52, card(1), card(1, 0, 1)], 1):
        assert all(r < 13 for c, r in zip(m.cards, m.claim_ranks) if is_wildcard(c, 1))


def test_bomb_hierarchy():
    ordered = [Move(BOMB, 10, [1]*4, []), Move(BOMB, 0, [1]*5, []),
               Move(SFLUSH, 10, [1]*5, []), Move(BOMB, 0, [1]*6, []),
               Move(BOMB, 0, [1]*10, []), Move(ROCKET, 0, [1]*4, [])]
    for lower, higher in zip(ordered, ordered[1:]):
        assert beats(higher, lower, 1)
        assert not beats(lower, higher, 1)
    assert beats(ordered[0], Move(SINGLE, 14, [53], [BJ]), 1)


@pytest.mark.parametrize("kind,ranks", [(STRAIGHT, [0, 1, 2, 3, 4]),
                                       (PLATE, [0, 0, 1, 1, 2, 2]),
                                       (TUBE, [0, 0, 0, 1, 1, 1])])
def test_ace_low_sequences(kind, ranks):
    counts = Counter()
    hand = []
    for r in ranks:
        hand.append(card(r, counts[r]))
        counts[r] += 1
    if kind == STRAIGHT:
        hand[2] += 1
    assert any(m.type == kind and m.key == 1 for m in gen_moves(hand, 7))


def test_no_wraparound_sequences():
    hand = [card(r, i % 4) for i, r in enumerate([10, 11, 12, 0, 1])]
    assert not any(m.type in (STRAIGHT, SFLUSH) for m in gen_moves(hand, 5))


def test_full_house_comparison_ignores_pair():
    a = Move(FULL, 7, [0]*5, [7]*3+[0]*2)
    b = Move(FULL, 6, [0]*5, [6]*3+[BJ]*2)
    assert beats(a, b, 1)


def test_claims_round_trip_and_card_ownership():
    rng = random.Random(2026)
    deck = list(range(108))
    for _ in range(80):
        rng.shuffle(deck)
        hand, level = deck[:27], rng.randrange(13)
        for m in gen_moves(hand, level):
            assert len(m.cards) == len(set(m.cards))
            assert set(m.cards) <= set(hand)
            claimed = claim_ids(m)
            decoded = classify_claim(m.cards, claimed, level)
            assert (decoded.type, decoded.key) == (m.type, m.key)
            assert len(m.claim_ranks) == len(m.cards)
            assert all(rank_of(c) == rank_of(cc) or is_wildcard(c, level) for c, cc in zip(m.cards, claimed))


def test_tribute_excludes_wild_and_return_excludes_ace():
    assert forced_tribute_card([card(1), 53, card(0)], 1) == 53
    assert default_return_card([card(0), card(5), card(1)], 1) == card(5)


def test_single_and_double_resistance():
    for mode in [('single', 1, 0), ('double', 1, 0)]:
        hands = [[card(2)], [53, 107], [card(3)], [card(4)]]
        rnd = GuandanRound(1, deal=hands, tribute_mode=mode)
        rnd._do_tribute()
        assert rnd.resist and rnd.lead_player == 0 and not rnd.events


def test_double_tribute_split_jokers_resists():
    rnd = GuandanRound(1, deal=[[8], [53], [12], [107]], tribute_mode=('double', 1, 0))
    rnd._do_tribute()
    assert rnd.resist


def test_tribute_card_conservation():
    for seed in range(20):
        rnd = GuandanRound(1, random.Random(seed), ('double', 1, 0))
        rnd._do_tribute()
        assert sorted(c for hand in rnd.hands for c in hand) == list(range(108))
        assert [len(h) for h in rnd.hands] == [27] * 4


def test_catch_wind_and_team_double_finish():
    # 0 finishes with BJ; everyone passes, so 2 must receive the lead.
    hands = [[53], [card(2), card(5)], [card(3), card(3, 1)], [card(4), card(6)]]
    rnd = GuandanRound(1, deal=hands)
    gen = rnd.play_steps()
    obs = next(gen)
    assert obs['player'] == 2 and obs['lead'] is None
    pair = next(i for i, move in enumerate(obs['legal']) if move.type == PAIR)
    with pytest.raises(StopIteration) as result:
        gen.send(pair)
    rewards, ranking = result.value.value
    assert ranking[:2] == [0, 2] and rewards == [3, -3, 3, -3]


def test_finish_owner_can_be_overcalled_before_catch_wind():
    hands = [[card(2)], [card(3), card(5)], [card(4), card(6)], [card(7), card(8)]]
    gen = GuandanRound(1, deal=hands).play_steps()
    obs = next(gen)
    assert obs['player'] == 1 and obs['lead_owner'] == 0
    assert any(m.type != PASS for m in obs['legal'])


def test_round_invariants_across_all_levels():
    from fabledan.agents import RandomAgent
    for level in range(13):
        for seed in range(3):
            agents = [RandomAgent(seed*4+i) for i in range(4)]
            rewards, ranking, rnd = play_round(agents, random.Random(seed+level*100), level)
            actual = [c for ev in rnd.events if ev[0] == 'play' for c in ev[2].cards]
            actual.extend(c for h in rnd.hands for c in h)
            assert sorted(actual) == list(range(108))
            assert sorted(ranking) == [0, 1, 2, 3]
            assert sum(rewards) == 0 and rewards[0] == rewards[2] and rewards[1] == rewards[3]


def test_illegal_action_and_deal_fail_closed():
    with pytest.raises(ValueError):
        GuandanRound(1, deal=[[0], [0], [1], [2]])
    gen = GuandanRound(1, random.Random(42)).play_steps()
    next(gen)
    with pytest.raises(ValueError, match="illegal"):
        gen.send(-1)


def test_team_rule_passes_on_partner():
    from fabledan.combos import PASS_MOVE
    lead = Move(SINGLE, 5, [card(5)], [5])
    obs = dict(player=0, level=1, hand=[card(7), card(8)], left=[2, 12, 10, 15],
               lead_owner=2, lead=lead, legal=[PASS_MOVE, Move(SINGLE, 8, [card(8)], [8])])
    assert TeamRuleAgent().act(obs) == 0
