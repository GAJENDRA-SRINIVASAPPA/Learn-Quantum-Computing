"""Quantum Mentor: a tiny local AI tutor for beginner quantum circuits."""

from __future__ import annotations

import html
import json
import math
import os
import urllib.parse
import urllib.request
from collections import defaultdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SQRT_HALF = 1 / math.sqrt(2)


def apply_single_qubit(state: list[complex], gate: str, qubit: int, qubits: int) -> list[complex]:
    matrix = {
        "H": ((SQRT_HALF, SQRT_HALF), (SQRT_HALF, -SQRT_HALF)),
        "X": ((0, 1), (1, 0)),
        "Z": ((1, 0), (0, -1)),
    }[gate]
    next_state = state[:]
    bit = 1 << (qubits - qubit - 1)
    for index in range(len(state)):
        if index & bit:
            continue
        partner = index | bit
        zero_value, one_value = state[index], state[partner]
        next_state[index] = matrix[0][0] * zero_value + matrix[0][1] * one_value
        next_state[partner] = matrix[1][0] * zero_value + matrix[1][1] * one_value
    return next_state


def apply_cnot(state: list[complex], control: int, target: int, qubits: int) -> list[complex]:
    next_state = state[:]
    control_bit = 1 << (qubits - control - 1)
    target_bit = 1 << (qubits - target - 1)
    for index in range(len(state)):
        if index & control_bit and not index & target_bit:
            partner = index | target_bit
            next_state[index], next_state[partner] = state[partner], state[index]
    return next_state


def parse_circuit(source: str, qubits: int) -> list[tuple[str, list[int]]]:
    operations = []
    for line_number, line in enumerate(source.splitlines(), 1):
        parts = line.strip().upper().split()
        if not parts:
            continue
        gate = parts[0]
        expected = 2 if gate != "CNOT" else 3
        if gate not in {"H", "X", "Z", "CNOT"} or len(parts) != expected:
            raise ValueError(f"Line {line_number}: use H, X, Z, or CNOT with qubit numbers.")
        try:
            indices = [int(value) for value in parts[1:]]
        except ValueError as error:
            raise ValueError(f"Line {line_number}: qubit numbers must be integers.") from error
        if any(index < 0 or index >= qubits for index in indices) or (gate == "CNOT" and indices[0] == indices[1]):
            raise ValueError(f"Line {line_number}: qubit numbers must be distinct values from 0 to {qubits - 1}.")
        operations.append((gate, indices))
    if not operations:
        raise ValueError("Add at least one gate to the circuit.")
    return operations


def simulate(source: str, qubits: int = 2) -> dict[str, Any]:
    if not 1 <= qubits <= 3:
        raise ValueError("This demo supports one to three qubits.")
    state = [0j] * (2**qubits)
    state[0] = 1 + 0j
    operations = parse_circuit(source, qubits)
    for gate, indices in operations:
        if gate == "CNOT":
            state = apply_cnot(state, indices[0], indices[1], qubits)
        else:
            state = apply_single_qubit(state, gate, indices[0], qubits)
    probabilities = {
        format(index, f"0{qubits}b"): round(abs(amplitude) ** 2, 4)
        for index, amplitude in enumerate(state)
        if abs(amplitude) ** 2 > 0.0001
    }
    return {"qubits": qubits, "operations": operations, "probabilities": probabilities}


def local_explanation(result: dict[str, Any]) -> str:
    probabilities = result["probabilities"]
    labels = list(probabilities)
    if len(labels) == 1:
        outcome = labels[0]
        if outcome == "0" * result["qubits"]:
            return "The circuit leaves the register in |0...0>. Every measurement returns all zeros."
        return f"The gates transform the starting state into |{outcome}>. A measurement returns {outcome} every time."
    if set(labels) == {"00", "11"}:
        return "The H followed by CNOT pattern creates a Bell state: the qubits are correlated, so measurements match even though each individual bit is unpredictable."
    largest = max(probabilities, key=probabilities.get)
    return f"The register is in a superposition of {len(labels)} outcomes. The most likely measurement is {largest} at {probabilities[largest] * 100:.1f}%. Measurement samples one outcome; it does not reveal every amplitude at once."


def ai_explanation(result: dict[str, Any]) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return local_explanation(result)
    payload = json.dumps({
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        "messages": [{"role": "system", "content": "Explain quantum circuits to a curious beginner in two concise sentences."}, {"role": "user", "content": json.dumps(result)}],
        "temperature": 0.2,
    }).encode()
    request = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            body = json.loads(response.read())
        return body["choices"][0]["message"]["content"]
    except (OSError, KeyError, IndexError, json.JSONDecodeError):
        return local_explanation(result)


HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Quantum Mentor</title>
<style>
:root{font-family:Georgia,serif;color:#17212b;background:#f4f0e8}*{box-sizing:border-box}body{margin:0}.shell{max-width:980px;margin:auto;padding:48px 24px}.eyebrow{font:700 12px system-ui;letter-spacing:.14em;text-transform:uppercase;color:#b34b2d}.hero{display:flex;justify-content:space-between;gap:24px;align-items:end;border-bottom:1px solid #c9c0b3;padding-bottom:30px}.hero h1{font-size:clamp(42px,7vw,78px);line-height:.94;margin:10px 0 0;max-width:650px;font-weight:500}.hero p{font:16px system-ui;line-height:1.5;max-width:280px;color:#52606b}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 1fr;gap:20px;margin-top:28px}.panel{background:#fffdf9;border:1px solid #d9d0c4;padding:22px;box-shadow:5px 5px 0 #e3d9cc}.panel h2{font-size:23px;font-weight:500;margin:0 0 8px}.hint{font:13px system-ui;color:#68747c;line-height:1.5}.field{display:flex;gap:10px;align-items:center;margin:18px 0;font:14px system-ui}.field input{width:60px;padding:9px;border:1px solid #bdb2a5;background:#fffdf9;font:inherit}.field input:focus,textarea:focus{outline:2px solid #d67b5d;outline-offset:2px}textarea{width:100%;height:190px;resize:vertical;padding:14px;border:1px solid #bdb2a5;background:#fbf8f1;font:16px monospace;line-height:1.7}button{border:0;background:#b34b2d;color:white;padding:13px 20px;font:700 13px system-ui;cursor:pointer;text-transform:uppercase;letter-spacing:.08em}button:hover{background:#87351f}.result{min-height:330px}.result.empty{display:grid;place-items:center;text-align:center;color:#7a827f;font:15px system-ui}.result.error{color:#a42f20;font:15px system-ui}.stats{display:flex;flex-wrap:wrap;gap:9px;margin:20px 0}.chip{background:#e9f0ed;padding:8px 11px;font:700 14px monospace}.bar{margin:13px 0;font:14px system-ui}.barhead{display:flex;justify-content:space-between;margin-bottom:5px}.track{height:10px;background:#e6ded4}.fill{height:100%;background:#3d7770}.explanation{border-top:1px solid #d9d0c4;margin-top:22px;padding-top:17px;font-size:18px;line-height:1.45}.footer{font:12px system-ui;color:#657078;margin-top:26px}@media(max-width:700px){.shell{padding:30px 16px}.hero{display:block}.hero p{max-width:none}.workspace{grid-template-columns:1fr}.panel{box-shadow:3px 3px 0 #e3d9cc}}
</style></head><body><main class="shell"><header class="hero"><div><div class="eyebrow">Learn quantum computing by seeing it</div><h1>Quantum Mentor</h1></div><p>Turn a few gates into a measurement you can actually explain.</p></header><section class="workspace"><form class="panel" method="post"><h2>Build a circuit</h2><p class="hint">Start in |00>. One gate per line. Qubit 0 is the leftmost bit.</p><label class="field">Qubits <input name="qubits" type="number" min="1" max="3" value="{{qubits}}"></label><textarea name="circuit" spellcheck="false">{{circuit}}</textarea><p class="hint">Examples: <code>H 0</code> &nbsp; <code>CNOT 0 1</code> &nbsp; <code>X 1</code></p><button type="submit">Explain circuit</button></form><section class="panel result {{result_class}}">{{result}}</section></section><div class="footer">Local simulator, deterministic fallback tutor. Set OPENAI_API_KEY to add an optional LLM explanation.</div></main></body></html>"""


def render_result(result: dict[str, Any] | None, error: str | None) -> tuple[str, str]:
    if error:
        return f"<p><strong>Let’s fix that:</strong> {html.escape(error)}</p>", "error"
    if not result:
        return "<p>Run a circuit and your probability distribution will appear here.</p>", "empty"
    bars = "".join(f'<div class="bar"><div class="barhead"><span>|{label}></span><strong>{probability * 100:.1f}%</strong></div><div class="track"><div class="fill" style="width:{probability * 100}%"></div></div></div>' for label, probability in result["probabilities"].items())
    explanation = html.escape(ai_explanation(result))
    return f'<h2>What will you measure?</h2><div class="stats"><span class="chip">{len(result["operations"])} gates</span><span class="chip">{result["qubits"]} qubits</span></div>{bars}<div class="explanation"><strong>Mentor says:</strong><br>{explanation}</div>', ""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.respond(None, None, "H 0\nCNOT 0 1")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        fields = urllib.parse.parse_qs(self.rfile.read(length).decode())
        circuit = fields.get("circuit", [""])[0]
        try:
            qubits = int(fields.get("qubits", ["2"])[0])
            result = simulate(circuit, qubits)
            self.respond(result, None, circuit, qubits)
        except (ValueError, KeyError) as error:
            self.respond(None, str(error), circuit, fields.get("qubits", ["2"])[0])

    def respond(self, result: dict[str, Any] | None, error: str | None, circuit: str, qubits: int | str = 2) -> None:
        rendered, result_class = render_result(result, error)
        page = HTML.replace("{{qubits}}", str(qubits)).replace("{{circuit}}", circuit).replace("{{result_class}}", result_class).replace("{{result}}", rendered)
        body = page.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Quantum Mentor running at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
