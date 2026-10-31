// Booking detection in the agent transcript (crosstalk/web/app.js).
//     node tests/booking_detection.test.js
const fs = require("fs");
const path = require("path");

const web = path.join(__dirname, "..", "crosstalk", "web");
const src = fs.readFileSync(path.join(web, "app.js"), "utf8");
const grab = (re) => src.match(re)[0];
const code = [grab(/const DAYS = .*\n.*HOUR_WORDS.*\n/), grab(/function detectBooking[\s\S]*?\n}\n/)].join("\n");
const listings = JSON.parse(fs.readFileSync(path.join(web, "..", "data", "listings.json")))
  .listings.map((l) => ({ ...l, free_slots: l.slots }));
const detect = new Function("state", `${code}; return detectBooking;`)({ listings: { listings } });

const cases = [
  ["Great, so I have you down for the Maple Street house on Saturday at 10 AM.", "maple|Saturday 10 AM"],
  ["Sure! Cedar Court on Sunday at three p.m. works.", "cedar|Sunday 3 PM"],
  ["The Willow Lane home has a pool. Do you want to see it?", null],
  ["Maple has Saturday at 2 or Sunday at 11. Okay, Sunday at eleven it is.", "maple|Sunday 11 AM"],
  ["I can book 18 Cedar Court at 5 PM on Friday for you.", "cedar|Friday 5 PM"],
  ["Harbor View is 359,000. It has a gym.", null],
  ["Sam, you're all set for the 11 o’clock viewing on Saturday at 18 Cedar Court.", "cedar|Saturday 11 AM"],
  // Real transcript: the model mixed up listings; Willow has no Saturday 11 slot.
  ["Sam, you allre set for the 11 o’clock viewing on Saturday with 23 Willow Lane.", null],
];
let failed = 0;
for (const [text, want] of cases) {
  const m = detect(text);
  const got = m ? `${m.listing.id}|${m.slot}` : null;
  if (got !== want) {
    failed++;
    console.log(`FAIL ${JSON.stringify(text)}: got ${got}, want ${want}`);
  }
}
console.log(`${cases.length - failed}/${cases.length} booking detection cases passed`);
process.exit(failed ? 1 : 0);
