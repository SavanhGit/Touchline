import pandas as pd

from soccer_tactics.analytics.sequences import segment_possessions, summarize_transitions


def ev(rows):
    d=dict(match_id="m1",period=1,team="home",event="pass",success=True,frame=0,time_s=0.,x0=0.,y0=0.,x1=1.,y1=0.)
    return pd.DataFrame([{**d,**r} for r in rows])


def test_possession_counts_and_terminal_reason_hand_calculation():
    e=ev([{"event_id":"p1"},{"event_id":"p2","time_s":2},{"event_id":"s","event":"shot","time_s":4,"success":False}])
    m=pd.DataFrame([{"event_id":"p1","entry":True,"success":True},{"event_id":"p2","entry":False,"success":True}])
    r=segment_possessions(e,m).iloc[0]
    assert r.event_ids==["p1","p2","s"] and (r.passes,r.completed_passes,r.final_third_entries,r.shots,r.goals)==(2,2,1,1,0)
    assert r.terminal_reason=="shot" and r.duration_s==4


def test_failed_pass_and_successful_defense_split_possessions():
    e=ev([{"event_id":"bad","success":False},{"event_id":"win","team":"away","event":"tackle","time_s":1},{"event_id":"a","team":"away","time_s":2},{"event_id":"int","event":"interception","time_s":3}])
    r=segment_possessions(e,pd.DataFrame())
    assert [x.event_ids for x in r.itertuples()]==[["bad"],["win","a"],["int"]]


def test_same_frame_order_is_stable():
    e=ev([{"event_id":"z","frame":3},{"event_id":"a","frame":3,"time_s":1}])
    assert segment_possessions(e,pd.DataFrame()).iloc[0].event_ids==["z","a"]


def test_possessions_never_cross_period_or_match():
    e=ev([{"event_id":"a"},{"event_id":"b","period":2},{"event_id":"c","match_id":"m2"}])
    assert len(segment_possessions(e,pd.DataFrame()))==3


def test_empty_events_return_empty_tables():
    e=ev([])
    assert segment_possessions(e,pd.DataFrame()).empty and summarize_transitions(e,pd.DataFrame(),pd.DataFrame()).empty


def test_transition_includes_exact_window_boundary():
    e=ev([{"event_id":"loss","success":False},{"event_id":"opp","team":"away","time_s":10}])
    assert summarize_transitions(e,segment_possessions(e,pd.DataFrame()),pd.DataFrame()).iloc[0].event_ids==["opp"]


def test_transition_excludes_event_after_window():
    e=ev([{"event_id":"loss","success":False},{"event_id":"opp","team":"away","time_s":10.01}])
    assert summarize_transitions(e,segment_possessions(e,pd.DataFrame()),pd.DataFrame()).empty


def test_transition_starts_at_successful_defensive_action():
    e=ev([{"event_id":"win","event":"tackle"},{"event_id":"next","time_s":1}])
    t=summarize_transitions(e,segment_possessions(e,pd.DataFrame()),pd.DataFrame())
    assert len(t)==1 and t.iloc[0].source_turnover_event_id=="win" and t.iloc[0].event_ids==["win","next"]


def test_no_transition_without_observed_opponent_event():
    e=ev([{"event_id":"win","event":"recovery"}])
    assert summarize_transitions(e,segment_possessions(e,pd.DataFrame()),pd.DataFrame()).empty


def test_transition_never_crosses_period():
    e=ev([{"event_id":"loss","success":False},{"event_id":"next","team":"away","period":2,"time_s":1}])
    assert summarize_transitions(e,segment_possessions(e,pd.DataFrame()),pd.DataFrame()).empty
