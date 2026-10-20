# Featured analysis: Metrica game 1, away team

Home 3–0 Away. The away team lost.

These patterns may have contributed; this report does not establish causation or promise that a suggested change would alter the result.

xT is trained on game 2 only. Teams and player IDs are anonymous. Data: [Metrica Sports](https://github.com/metrica-sports/sample-data).

## Buildup losses on the left

22 of 76 buildup passes in this channel were lost (29%); total modeled cost 0.288 xT. Median nearest-opponent distance was 3.2 m.

**Adjustment to investigate:** Investigate offering a closer passing option on the left, or varying the buildup route.

**Tradeoff:** Bringing support toward the ball can leave less space occupied further forward.

| Evidence ID | Period | Frame | Source seconds | Player |
|---|---:|---:|---:|---|
| metrica_1:1:57624:402 | 1 | 57624 | 2304.96 | Player16 |
| metrica_1:1:31738:258 | 1 | 31738 | 1269.52 | Player20 |
| metrica_1:1:69030:494 | 1 | 69030 | 2761.20 | Player20 |

## Limited direct entries from the centre

9 of 137 passes starting outside the final third crossed into it successfully (7%). Other passes may be useful circulation; this does not establish poor decisions.

**Adjustment to investigate:** Review the supporting moments for a forward passing option, and compare routes through the other channels.

**Tradeoff:** More direct progression can increase turnover exposure; low entry rate alone does not justify forcing a pass.

| Evidence ID | Period | Frame | Source seconds | Player |
|---|---:|---:|---:|---|
| metrica_1:1:21580:168 | 1 | 21580 | 863.20 | Player16 |
| metrica_1:1:67514:475 | 1 | 67514 | 2700.56 | Player25 |
| metrica_1:2:113498:799 | 2 | 113498 | 4539.92 | Player21 |

## Opponent danger through their centre

13 of 18 opponent shots came through their attacking centre channel (72%). Channels are measured in the opponent's attacking direction.

**Adjustment to investigate:** Investigate earlier pressure or additional cover in the opponent's centre attacking channel.

**Tradeoff:** Extra cover in one channel can free space elsewhere; these counts do not identify the responsible defender.

| Evidence ID | Period | Frame | Source seconds | Player |
|---|---:|---:|---:|---|
| metrica_1:1:2289:23 | 1 | 2289 | 91.56 | Player9 |
| metrica_1:2:99032:707 | 2 | 99032 | 3961.28 | Player9 |
| metrica_1:1:9628:77 | 1 | 9628 | 385.12 | Player8 |

## Verification and interpretation

The source audit compared all displayed event fields, including type, player, team, period, source seconds and coordinates, with the original event CSV. Event-to-player tracking coverage is 100.0%; median event alignment is 0.331 m and p90 alignment is 0.506 m. See `release-validation.json` for all four team reports and measured quality results.

Counts exclude carries. Shot-pattern findings exclude penalties, while the result includes every recorded goal. Turnover cost and player detriment are exploratory xT quantities. Pitch control is a simplified spatial model. Channel concentrations describe locations, not defensive responsibility.
