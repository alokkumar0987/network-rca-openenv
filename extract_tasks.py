#!/usr/bin/env python
"""
Extract real network incidents from the database and generate data/tasks.json
with added metrics and logs for deeper reasoning.
"""
import mysql.connector
import pandas as pd
import json
import os
import random
from collections import defaultdict
import networkx as nx

random.seed(42)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'Alok@12&kumar#@',
    'database': 'railtel'
}

def connect_db():
    return mysql.connector.connect(**DB_CONFIG)

def build_topology(conn):
    ne_df = pd.read_sql("SELECT ID, NE_NAME FROM network_element", conn)
    ne_id_to_name = dict(zip(ne_df['ID'], ne_df['NE_NAME']))

    G = nx.Graph()
    for table in ['bgp_link', 'ospf_link', 'lldp_link', 'isis_link']:
        try:
            df = pd.read_sql(f"SELECT SOURCE_NE_ID, DESTINATION_NE_ID FROM {table}", conn)
            for _, row in df.iterrows():
                src = ne_id_to_name.get(row['SOURCE_NE_ID'])
                dst = ne_id_to_name.get(row['DESTINATION_NE_ID'])
                if src and dst and src != dst:
                    G.add_edge(src, dst)
        except:
            pass
    return G, ne_id_to_name

def get_root_cause(incident_alarms, alarm_library):
    root_alarm = incident_alarms[
        incident_alarms['CORRELATION_TYPE'].astype(str).str.upper().str.contains('ROOT', na=False)
    ]
    if not root_alarm.empty:
        root_alarm = root_alarm.iloc[0]
        root_device = root_alarm['ENTITY_NAME']
        code = root_alarm['ALARM_CODE']
        cause_row = alarm_library[alarm_library['ALARM_IDENTIFIER'] == code]
        if not cause_row.empty:
            cause = cause_row.iloc[0].get('PROBABLE_CAUSE', 'Unknown cause')
        else:
            cause = root_alarm.get('PROBABLE_CAUSE', 'Unknown cause')
        return cause, root_device

    earliest = incident_alarms.sort_values('OPEN_TIME').iloc[0]
    root_device = earliest['ENTITY_NAME']
    cause = earliest.get('PROBABLE_CAUSE', 'Unknown cause')
    return cause, root_device

def generate_metrics(devices, root_device):
    metrics = {}
    for dev in devices:
        if dev == root_device:
            metrics[dev] = {
                'latency_ms': random.randint(150, 500),
                'packet_loss_pct': round(random.uniform(5, 20), 1),
                'cpu_util_pct': random.randint(70, 95)
            }
        else:
            metrics[dev] = {
                'latency_ms': random.randint(10, 50),
                'packet_loss_pct': round(random.uniform(0, 1), 1),
                'cpu_util_pct': random.randint(20, 60)
            }
    return metrics

def generate_logs(devices, root_device):
    logs = {}
    for dev in devices:
        if dev == root_device:
            logs[dev] = [
                f"2025-04-01T10:23:15 ERROR: Interface flapping on {dev}",
                f"2025-04-01T10:23:20 ERROR: BGP session down with neighbor",
                f"2025-04-01T10:23:30 CRITICAL: Link to {next((d for d in devices if d != dev), 'unknown')} down"
            ]
        else:
            logs[dev] = [
                f"2025-04-01T10:22:00 INFO: Normal operation",
                f"2025-04-01T10:23:18 WARNING: Increased latency detected"
            ]
    return logs

def extract_incidents(conn, G, ne_id_to_name, limit_per_difficulty=3):
    alarm_cols = [
        'INCIDENT_ID', 'OPEN_TIME', 'ALARM_CODE', 'ALARM_NAME', 'SEVERITY',
        'ENTITY_NAME', 'CORRELATION_TYPE', 'PROBABLE_CAUSE', 'DESCRIPTION'
    ]
    alarm_df = pd.read_sql(f"SELECT {','.join(alarm_cols)} FROM alarm", conn)
    alarm_df = alarm_df.dropna(subset=['INCIDENT_ID', 'ENTITY_NAME'])
    alarm_df['INCIDENT_ID'] = alarm_df['INCIDENT_ID'].astype(str)
    alarm_df['OPEN_TIME'] = pd.to_datetime(alarm_df['OPEN_TIME'], errors='coerce')
    alarm_df = alarm_df.sort_values('OPEN_TIME')

    alarm_library = pd.read_sql("SELECT ALARM_IDENTIFIER, PROBABLE_CAUSE FROM alarm_library", conn)

    incidents = defaultdict(list)
    for _, row in alarm_df.iterrows():
        incidents[row['INCIDENT_ID']].append(row)

    easy, medium, hard = [], [], []
    for inc_id, rows in incidents.items():
        num_alarms = len(rows)
        devices = set(r['ENTITY_NAME'] for r in rows)
        root_cause, root_device = get_root_cause(pd.DataFrame(rows), alarm_library)

        if num_alarms <= 2:
            difficulty = 'easy'
        elif num_alarms <= 4 and len(devices) <= 2:
            difficulty = 'medium'
        else:
            difficulty = 'hard'

        alarms = []
        detailed_descriptions = {}
        for r in rows:
            alarm_id = f"{inc_id}_{r['OPEN_TIME'].timestamp()}"
            short_desc = r['ALARM_NAME']
            long_desc = r['DESCRIPTION'] or r['ALARM_NAME']
            alarms.append({
                'id': alarm_id,
                'code': r['ALARM_CODE'],
                'name': r['ALARM_NAME'],
                'severity': r['SEVERITY'],
                'device': r['ENTITY_NAME'],
                'description': short_desc
            })
            detailed_descriptions[alarm_id] = long_desc

        relevant_ids = {a['id'] for a in alarms}
        incident_edges = []
        for u, v in G.edges():
            if u in devices or v in devices:
                incident_edges.append((u, v))

        metrics = generate_metrics(devices, root_device)
        logs = generate_logs(devices, root_device)

        task = {
            'alarms': alarms,
            'topology_edges': incident_edges,
            'relevant_alarm_ids': list(relevant_ids),
            'dependency_alarms': [],
            'ground_truth': root_cause,
            'max_steps': max(10, num_alarms * 2),
            'description': f"Real incident {inc_id} with {num_alarms} alarms",
            'future_alarms': [],
            'detailed_descriptions': detailed_descriptions,
            'metrics': metrics,
            'logs': logs
        }

        if difficulty == 'easy':
            easy.append(task)
        elif difficulty == 'medium':
            medium.append(task)
        else:
            hard.append(task)

    return easy[:limit_per_difficulty], medium[:limit_per_difficulty], hard[:limit_per_difficulty]

def main():
    conn = connect_db()
    G, ne_id_to_name = build_topology(conn)
    easy, medium, hard = extract_incidents(conn, G, ne_id_to_name, limit_per_difficulty=3)

    tasks = {
        'easy': easy,
        'medium': medium,
        'hard': hard
    }

    os.makedirs('data', exist_ok=True)
    with open('data/tasks.json', 'w') as f:
        json.dump(tasks, f, indent=2, default=str)

    print(f"Extracted {len(easy)} easy, {len(medium)} medium, {len(hard)} hard tasks")
    conn.close()

if __name__ == '__main__':
    main()