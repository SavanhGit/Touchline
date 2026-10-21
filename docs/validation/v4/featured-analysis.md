# Featured analysis: Metrica game 1, away team

Home 3–0 Away. The away team lost.

These patterns may have contributed; this report does not establish causation or promise that a suggested change would alter the result.

xT is trained on games 2 and 3 only. Teams and player IDs are anonymous. Data: [Metrica Sports](https://github.com/metrica-sports/sample-data).

## Limited direct entries from the centre

9 of 137 passes starting outside the final third crossed into it successfully (7%). Other passes may be useful circulation; this does not establish poor decisions.

**Adjustment to investigate:** Review the supporting moments for a forward passing option, and compare routes through the other channels.

**Tradeoff:** More direct progression can increase turnover exposure; low entry rate alone does not justify forcing a pass.

| Evidence ID | Period | Frame | Source seconds | Player |
|---|---:|---:|---:|---|
| metrica_1:1:21580:213 | 1 | 21580 | 863.20 | Player16 |
| metrica_1:1:67514:584 | 1 | 67514 | 2700.56 | Player25 |
| metrica_1:2:113498:979 | 2 | 113498 | 4539.92 | Player21 |

## Transitions without a direct entry or shot

91 of 103 observed transitions had no completed final-third entry or shot within the clipped 10-second window.

**Adjustment to investigate:** Review forward support at these transition starts as a tactical hypothesis.

**Tradeoff:** Sending support forward can reduce cover after another turnover; these observations do not establish causality.

| Evidence ID | Period | Frame | Source seconds | Player |
|---|---:|---:|---:|---|
| metrica_1:1:15222:145 | 1 | 15222 | 608.88 | Player17 |
| metrica_1:1:66901:576 | 1 | 66901 | 2676.04 | Player22 |
| metrica_1:2:115892:1009 | 2 | 115892 | 4635.68 | Player15 |

## Buildup losses on the left

22 of 76 buildup passes in this channel were lost (29%); total modeled cost 0.313 xT. Median nearest-opponent distance was 3.2 m.

**Adjustment to investigate:** Investigate offering a closer passing option on the left, or varying the buildup route.

**Tradeoff:** Bringing support toward the ball can leave less space occupied further forward.

| Evidence ID | Period | Frame | Source seconds | Player |
|---|---:|---:|---:|---|
| metrica_1:1:57624:498 | 1 | 57624 | 2304.96 | Player16 |
| metrica_1:1:31738:319 | 1 | 31738 | 1269.52 | Player20 |
| metrica_1:1:22660:218 | 1 | 22660 | 906.40 | Player22 |

## Verification and interpretation

The source audit compared all displayed event fields, including type, player, team, period, source seconds and coordinates, with the original event source. Event-to-player tracking coverage is 100.0%; median event alignment is 0.333 m and p90 alignment is 0.507 m. See `release-validation.json` for all six team reports and measured quality results.

Counts exclude carries. Shot-pattern findings exclude penalties, while the result includes every recorded goal. Turnover cost and player detriment are exploratory xT quantities. Pitch control is a simplified spatial model. Channel concentrations describe locations, not defensive responsibility.
