# SideQuest Nugen travel-impact model

This is a synthetic training corpus for the SideQuest What If feature. It
contains no user accounts, saved itineraries, exact trip dates, coordinates,
or credentials.

## Task

Use the supplied route, forecast summary, itinerary schedule, stop safety
context and the traveller's chosen scenario to explain possible impacts.
Return exactly one JSON object with this schema:

```json
{
  "accessibility": "likely_open | limited | unknown",
  "delay_prob": null,
  "demand_shift_pct": null,
  "safety_impact": "low | moderate | high | unknown",
  "cascades": [],
  "recommendation": "continue_with_caution | wait_for_update | consider_alternative | insufficient_data",
  "confidence": "low | medium | high",
  "guidance": "A concise explanation for the traveller."
}
```

`delay_prob` is a 0–1 estimate that the supplied scenario adds a meaningful
delay beyond SideQuest's baseline schedule. `demand_shift_pct` is an estimated
change in visitor interest relative to normal conditions, from -100 to 100.
These estimates are not calibrated forecasts. Use `null` when the inputs do not
support a useful estimate and use low confidence for sparse or generalized
weather data. The examples' numbers are illustrative scenario labels, not
measured outcomes.

Never claim a live flood, road closure, open facility, or guaranteed safety.
Do not invent nearby hospitals, police stations, phone numbers, road conditions,
or place details. “Likely open” means only that the supplied inputs include no
known restriction; it is not a verified operational status. If there is no
current forecast or local alert, say so and recommend checking official local
sources. Keep route timing and stop selection consistent with SideQuest's
calculated schedule. Put at most three concrete itinerary effects in
`cascades`; use an empty list if none are supported by the supplied facts.

## Demonstrations

### Mild rain, schedule still fits
Input: Scenario: light rain. Route forecast: 25% precipitation chance, up to
0.5 mm/hour. No local closure or flood alert was supplied. All 2 saved stops
fit with 35 minutes of schedule slack. Place safety labels are not available.
Output: {"accessibility":"likely_open","delay_prob":0.08,"demand_shift_pct":-3,"safety_impact":"low","cascades":["Both saved stops still fit the current schedule."],"recommendation":"continue_with_caution","confidence":"low","guidance":"The forecast summary suggests light rain and the schedule still has 35 minutes of slack. No live road or flood alert was supplied, so check official local updates before departure."}

### Heavy rain with a tight schedule
Input: Scenario: heavy rain. Forecast along sampled route: 75% precipitation
chance, up to 8 mm/hour. A local flood alert is not available. After the
traveller's 30-minute delay assumption, one stop no longer fits and 5 minutes
of slack remain.
Output: {"accessibility":"limited","delay_prob":0.62,"demand_shift_pct":-25,"safety_impact":"moderate","cascades":["One saved stop no longer fits after the 30-minute delay assumption.","Only 5 minutes of schedule slack remain."],"recommendation":"wait_for_update","confidence":"low","guidance":"Heavy rain is forecast in the supplied route summary and the revised plan has only 5 minutes of slack. This does not confirm flooding or a closure. Check official weather and road updates before deciding."}

### No current forecast
Input: Scenario: heavy rain. No route forecast, current alert, closure report,
or weather measurements were supplied. Baseline itinerary timing is available.
Output: {"accessibility":"unknown","delay_prob":null,"demand_shift_pct":null,"safety_impact":"unknown","cascades":[],"recommendation":"insufficient_data","confidence":"low","guidance":"There is no current route forecast or local alert in the supplied data, so I cannot assess weather access or safety. Check official local weather and road sources."}

### Severe forecast, no official local alert
Input: Scenario: high winds and heavy rain. Route forecast: 90% precipitation
chance and 18 mm/hour. No official road closure or emergency alert was supplied.
The route contains an outdoor viewpoint and all other stops fit the schedule.
Output: {"accessibility":"limited","delay_prob":0.78,"demand_shift_pct":-35,"safety_impact":"high","cascades":["Outdoor-viewpoint conditions may be affected by the supplied severe-weather forecast.","Other stops still fit the calculated schedule."],"recommendation":"wait_for_update","confidence":"low","guidance":"The supplied forecast indicates severe rain and wind, which may disrupt outdoor plans. No official closure or emergency alert was provided, so this is not a live safety determination. Check official local advisories before travelling."}

### Delay entered without weather information
Input: Scenario: traffic delay entered by the traveller, 45 minutes. Weather
forecast is clear. There is no live traffic feed. One of 3 planned stops no
longer fits after the delay and one SideQuest alternative is listed.
Output: {"accessibility":"unknown","delay_prob":null,"demand_shift_pct":null,"safety_impact":"unknown","cascades":["One planned stop no longer fits after the traveller-entered 45-minute delay.","One alternative is available from SideQuest's route results."],"recommendation":"consider_alternative","confidence":"low","guidance":"The 45-minute delay is a planning assumption, not a live traffic measurement. One stop no longer fits; review the listed alternative and check a live navigation service."}

### Stop safety context says caution
Input: Scenario: heavy rain. The route forecast is unavailable. One saved stop
is marked caution by SideQuest's safety data, and the nearest hospital is
listed as 18 km away. Data freshness and police-station distance are unknown.
Output: {"accessibility":"unknown","delay_prob":null,"demand_shift_pct":null,"safety_impact":"moderate","cascades":["The saved stop has a caution label in SideQuest's available safety data."],"recommendation":"insufficient_data","confidence":"low","guidance":"SideQuest's available data marks the stop caution and lists a hospital 18 km away. The forecast and data freshness are unknown, so this is not a live safety assessment. Verify current local conditions and services."}

### Later departure improves weather outlook
Input: The route forecast at planned departure has 70% rain probability. The
forecast two hours later has 30%. Shifting departure and arrive-by times by two
hours keeps every stop and leaves 40 minutes of slack. No road alert is known.
Output: {"accessibility":"likely_open","delay_prob":0.12,"demand_shift_pct":-5,"safety_impact":"low","cascades":["A two-hour later departure keeps every saved stop in the schedule.","The later sampled forecast has lower rain probability."],"recommendation":"consider_alternative","confidence":"low","guidance":"The later forecast summary shows lower rain probability, and the shifted schedule keeps every stop with 40 minutes to spare. Forecasts can change and no road alert was supplied; check official updates before leaving."}

### Time window cannot meet deadline
Input: Scenario: disruption adds 120 minutes by traveller assumption. Even
after SideQuest removes all detours, the calculated arrival is 55 minutes past
the deadline. There is no live traffic feed.
Output: {"accessibility":"unknown","delay_prob":null,"demand_shift_pct":null,"safety_impact":"unknown","cascades":["The direct schedule still arrives 55 minutes after the deadline.","All detours have already been removed from the scenario plan."],"recommendation":"consider_alternative","confidence":"low","guidance":"Under the 120-minute delay assumption, even the direct plan arrives 55 minutes after your deadline. This is not a live traffic prediction. Consider changing the trip timing and check a current navigation service."}

### No route or forecast facts
Input: The traveller asks “Is it safe to go now?” No route, date, forecast,
alerts, or safety information is available.
Output: {"accessibility":"unknown","delay_prob":null,"demand_shift_pct":null,"safety_impact":"unknown","cascades":[],"recommendation":"insufficient_data","confidence":"low","guidance":"I don't have a route, current forecast, or local safety information for this trip. I can't determine whether it is safe to go. Check official local weather, road, and emergency advisories."}

### Large delay with no replacement gem
Input: Scenario: traveller-entered 90-minute delay. Two planned stops are
removed. No replacement is returned and the revised direct route has 8 minutes
of slack. Live traffic and closure data are not connected.
Output: {"accessibility":"unknown","delay_prob":null,"demand_shift_pct":null,"safety_impact":"unknown","cascades":["Two planned stops no longer fit after the 90-minute assumption.","No replacement was returned and only 8 minutes of slack remain."],"recommendation":"consider_alternative","confidence":"low","guidance":"Two stops no longer fit under the 90-minute delay assumption, and there is not enough spare time for another stop. Live traffic and closure data are unavailable, so check current route conditions."}

