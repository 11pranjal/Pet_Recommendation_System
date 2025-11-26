document.getElementById("quizForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  // Must match FEATURE_COLUMNS order in backend/train.py
  const answers = [
    parseInt(document.getElementById("home_type").value),        // home_type
    parseInt(document.getElementById("activity_level").value),   // activity_level
    parseInt(document.getElementById("children").value),         // children
    parseInt(document.getElementById("other_pets").value),       // other_pets
    parseInt(document.getElementById("age_pref").value),         // age_pref
    parseInt(document.getElementById("size_pref").value),        // size_pref
    parseInt(document.getElementById("gender_pref").value),      // gender_pref
    parseInt(document.getElementById("grooming").value),         // grooming
    parseInt(document.getElementById("meat_diet").value),        // meat_diet
    parseInt(document.getElementById("vaccine_importance").value),// vaccine_importance
    parseInt(document.getElementById("health_acceptance").value) // health_acceptance
  ];

  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({answers})
  });
  const data = await res.json();
  if (data.ok) {
    document.getElementById("result").innerText = "We recommend: " + data.recommendation;
  } else {
    document.getElementById("result").innerText = "Error: " + (data.message || "unknown");
  }
});