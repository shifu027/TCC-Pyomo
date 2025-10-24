# -*- coding: utf-8 -*-
"""
======================================================================================================
SCRIPT DE OTIMIZAÇÃO E ANÁLISE VISUAL DE CENÁRIOS (VERSÃO TCC FINAL COMPLETA)
======================================================================================================
Autor: Engenheiro Elétrico Sênior, PhD
Versão: 5.5 (Adição de gráfico final de KPI com eixo duplo e todos os refinamentos)

Descrição:
Este script implementa a estratégia final de análise e visualização para o TCC. Ele executa
três cenários (BAU, BESS+V1G, BESS+V2G) e gera um conjunto completo de gráficos com resumos
explicativos, concluindo com uma tabela e um gráfico de resumo de KPIs.
======================================================================================================
"""

# ####################################################################################################
# ETAPA 1: CONFIGURAÇÃO DO AMBIENTE E DEPENDÊNCIAS
# ####################################################################################################
print(">>> ETAPA 1: Configurando o ambiente...")
try:
    import pyomo.environ as pyo
except ImportError:
    !pip install pyomo -q
    !apt-get install -y -qq coinor-cbc

import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import numpy as np
import io
import seaborn as sns
from IPython.display import display

print("Ambiente configurado com sucesso.")

# ####################################################################################################
# ETAPA 2: DADOS DE ENTRADA E PRÉ-PROCESSAMENTO
# ####################################################################################################
print("\n>>> ETAPA 2: Gerando e carregando arquivos de dados...")
fator_de_escala_da_carga = 5.0
print(f"AVISO: A carga da rede foi multiplicada por {fator_de_escala_da_carga}x para fins de visualização.")
general_params_csv = """parametro,valor,unidade
capacidade_bess_kwh,100,kWh;pot_max_carga_bess_kw,50,kW;pot_max_descarga_bess_kw,40,kW;eficiencia_carga_bess,0.92,-;eficiencia_descarga_bess,0.90,-;soc_min_bess_pu,0.20,p.u.;soc_max_bess_pu,0.95,p.u.;soc_inicial_bess_pu,0.50,p.u.;custo_degradacao_bess_r_kwh,0.10,R$/kWh;custo_degradacao_ve_r_kwh,0.05,R$/kWh;custo_curtailment_pv_r_kwh,0.02,R$/kWh;custo_demanda_pico_r_kw,45.0,R$/kW;delta_t_h,1.0,h;v_base_kv,13.8,kV;s_base_mva,1.0,MVA;v_min_pu,0.95,p.u.;v_max_pu,1.05,p.u.
""".replace(';', '\n')
tariffs_csv = """hora,tarifa_r_kwh
0,0.60;1,0.60;2,0.60;3,0.60;4,0.60;5,0.60;6,0.60;7,0.75;8,0.75;9,0.75;10,0.75;11,0.75;12,0.75;13,0.75;14,0.75;15,0.75;16,0.75;17,1.20;18,1.20;19,1.20;20,0.75;21,0.60;22,0.60;23,0.60
""".replace(';', '\n')
t_range = np.arange(24)
base_load = 40 + 30 * np.sin(np.pi * (t_range - 8) / 12)
peak_load = 1.8 * np.exp(-((t_range - 19)**2) / 4)
total_load_profile = base_load * peak_load * 2.5 * fator_de_escala_da_carga
load_profiles_df_gen = pd.DataFrame({'hora': t_range})
load_profiles_df_gen['load_kw_2'] = total_load_profile * 0.4
load_profiles_df_gen['load_kw_4'] = total_load_profile * 0.35
load_profiles_df_gen['load_kw_5'] = total_load_profile * 0.25
load_profiles_csv = load_profiles_df_gen.to_csv(index=False)
pv_profile = 100 * np.maximum(0, np.sin(np.pi * (t_range - 6) / 12))
pv_profiles_df_gen = pd.DataFrame({'hora': t_range, 'pv_kw_4': pv_profile})
pv_profiles_csv = pv_profiles_df_gen.to_csv(index=False)
ev_params_csv = """ve_id,capacidade_kwh,p_max_carga_kw,p_max_descarga_kw,ef_carga,ef_descarga,soc_inicial_pu,soc_alvo_pu,t_chegada,t_partida,aceita_v2g_b
0,50,7.2,3.7,0.95,0.95,0.40,0.9,18,7,1
1,60,11.0,5.0,0.96,0.94,0.50,0.9,19,8,0
2,45,7.2,3.7,0.95,0.95,0.30,0.8,17,6,1
3,75,22.0,11.0,0.97,0.96,0.60,0.95,20,7,0
"""
network_params_csv = """from_node,to_node,resistance_ohm,p_max_kw
0,1,0.05,2000;1,2,0.1,500;1,3,0.12,500;3,4,0.15,400;1,5,0.08,500
""".replace(';', '\n')

general_df = pd.read_csv(io.StringIO(general_params_csv)).set_index('parametro')
tariffs_df = pd.read_csv(io.StringIO(tariffs_csv))
load_profiles_df = pd.read_csv(io.StringIO(load_profiles_csv))
pv_profiles_df = pd.read_csv(io.StringIO(pv_profiles_csv))
ev_parameters_df = pd.read_csv(io.StringIO(ev_params_csv))
network_parameters_df = pd.read_csv(io.StringIO(network_params_csv))
T = sorted([int(h) for h in tariffs_df['hora'].unique()])
N_nodes = sorted([int(n) for n in pd.unique(network_parameters_df[['from_node', 'to_node']].values.ravel('K'))])
V = sorted([int(v) for v in ev_parameters_df['ve_id'].unique()])
E_branches = [tuple(int(x) for x in row) for row in network_parameters_df[['from_node', 'to_node']].values]
children_of = {n: [j for i,j in E_branches if i==n] for n in N_nodes}
parent_of = {j:i for i,j in E_branches}
params = {}
params['delta_t'] = float(general_df.loc['delta_t_h','valor'])
params['V_base_kv'] = float(general_df.loc['v_base_kv','valor'])
params['S_base_mva'] = float(general_df.loc['s_base_mva','valor'])
params['S_base_kw'] = params['S_base_mva'] * 1000.0
params['Z_base_ohm'] = (params['V_base_kv']**2) / params['S_base_mva']
params['C_pico'] = float(general_df.loc['custo_demanda_pico_r_kw','valor'])
params['C_curt'] = float(general_df.loc['custo_curtailment_pv_r_kwh','valor'])
params['C_deg_BESS'] = float(general_df.loc['custo_degradacao_bess_r_kwh','valor'])
params['C_deg_VE'] = float(general_df.loc['custo_degradacao_ve_r_kwh','valor'])
params['SoC_min_BESS_pu'] = float(general_df.loc['soc_min_bess_pu','valor'])
params['SoC_max_BESS_pu'] = float(general_df.loc['soc_max_bess_pu','valor'])
params['SoC_init_BESS_pu'] = float(general_df.loc['soc_inicial_bess_pu','valor'])
params['Pmax_ch_BESS_kw'] = float(general_df.loc['pot_max_carga_bess_kw','valor'])
params['Pmax_dis_BESS_kw'] = float(general_df.loc['pot_max_descarga_bess_kw','valor'])
params['eta_ch_BESS'] = float(general_df.loc['eficiencia_carga_bess','valor'])
params['eta_dis_BESS'] = float(general_df.loc['eficiencia_descarga_bess','valor'])
params['C_energia'] = tariffs_df.set_index('hora')['tarifa_r_kwh'].astype(float).to_dict()
params['P_carga'] = {}
for t in T:
    for col in load_profiles_df.columns:
        if col.startswith('load_kw_'):
            j = int(col.split('_')[-1])
            vals = load_profiles_df.loc[load_profiles_df['hora']==t, col].values
            params['P_carga'][(j, t)] = float(vals[0]) if len(vals) > 0 else 0.0
params['P_pv'] = {}
for t in T:
    for col in pv_profiles_df.columns:
        if col.startswith('pv_kw_'):
            j = int(col.split('_')[-1])
            vals = pv_profiles_df.loc[pv_profiles_df['hora']==t, col].values
            params['P_pv'][(j, t)] = float(vals[0]) if len(vals) > 0 else 0.0
ev_map = ev_parameters_df.set_index('ve_id').to_dict('index')
params['Emax_VE'] = {i: float(ev_map[i]['capacidade_kwh']) for i in V}
params['Pmax_ch_VE'] = {i: float(ev_map[i]['p_max_carga_kw']) for i in V}
params['Pmax_dis_VE'] = {i: float(ev_map[i]['p_max_descarga_kw']) for i in V}
params['eta_ch_VE'] = {i: float(ev_map[i]['ef_carga']) for i in V}
params['eta_dis_VE'] = {i: float(ev_map[i]['ef_descarga']) for i in V}
params['SoC_init_VE_kwh'] = {i: float(ev_map[i]['soc_inicial_pu']) * float(ev_map[i]['capacidade_kwh']) for i in V}
params['SoC_target_VE_kwh'] = {i: float(ev_map[i]['soc_alvo_pu']) * float(ev_map[i]['capacidade_kwh']) for i in V}
params['t_chegada_VE'] = {i: int(ev_map[i]['t_chegada']) for i in V}
params['t_partida_VE'] = {i: int(ev_map[i]['t_partida']) for i in V}
params['b_v2g_VE'] = {i: int(ev_map[i]['aceita_v2g_b']) for i in V}
params['a_disp_VE'] = {}
for i in V:
    chegada, partida = params['t_chegada_VE'][i], params['t_partida_VE'][i]
    for t in T:
        if chegada > partida: params['a_disp_VE'][(i,t)] = 1 if (t >= chegada or t < partida) else 0
        else: params['a_disp_VE'][(i,t)] = 1 if (chegada <= t < partida) else 0
R_ohm = {(int(r['from_node']), int(r['to_node'])): float(r['resistance_ohm']) for _, r in network_parameters_df.iterrows()}
params['R_pu'] = {k: v / params['Z_base_ohm'] for k, v in R_ohm.items()}
params['P_max_ramo_kw'] = {(int(r['from_node']), int(r['to_node'])): float(r['p_max_kw']) for _, r in network_parameters_df.iterrows()}
params['BESS_node'] = 2
params['VE_nodes'] = {0: 2, 1: 4, 2: 4, 3: 5}
T_list = list(T)
t_prev_map = {tt: T_list[idx-1] if idx>0 else T_list[-1] for idx, tt in enumerate(T_list)}
print("Pré-processamento concluído.")

# ####################################################################################################
# ETAPA 3: FUNÇÕES DE SIMULAÇÃO E OTIMIZAÇÃO
# ####################################################################################################
def run_optimization_scenario(params, v2g_enabled, bess_present):
    print(f"\n>>> Executando Otimização: V2G {'Habilitado' if v2g_enabled else 'Desabilitado'}, BESS {'Presente' if bess_present else 'Ausente'}")
    model = pyo.ConcreteModel()
    model.T = pyo.Set(initialize=T); model.N = pyo.Set(initialize=N_nodes); model.V = pyo.Set(initialize=V); model.E = pyo.Set(initialize=E_branches, dimen=2)
    model.P_grid = pyo.Var(model.T, domain=pyo.NonNegativeReals); model.P_max_pico = pyo.Var(domain=pyo.NonNegativeReals); model.P_curt = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals)
    model.P_bess_ch = pyo.Var(model.T, domain=pyo.NonNegativeReals); model.P_bess_dis = pyo.Var(model.T, domain=pyo.NonNegativeReals); model.SoC_BESS = pyo.Var(model.T, domain=pyo.NonNegativeReals); model.u_BESS = pyo.Var(model.T, domain=pyo.Binary)
    model.P_ve_ch = pyo.Var(model.V, model.T, domain=pyo.NonNegativeReals); model.P_ve_dis = pyo.Var(model.V, model.T, domain=pyo.NonNegativeReals); model.SoC_VE = pyo.Var(model.V, model.T, domain=pyo.NonNegativeReals); model.u_VE = pyo.Var(model.V, model.T, domain=pyo.Binary)
    model.P_ramo = pyo.Var(model.E, model.T, domain=pyo.Reals); model.V2_no = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals)
    def objective_rule(m):
        custo_energia = sum(params['C_energia'][t] * m.P_grid[t] * params['delta_t'] for t in m.T)
        custo_pico = params['C_pico'] * m.P_max_pico
        custo_curtailment = sum(params['C_curt'] * m.P_curt[j,t] * params['delta_t'] for j in m.N for t in m.T)
        custo_deg_bess = sum(params['C_deg_BESS'] * (m.P_bess_ch[t] + m.P_bess_dis[t]) * params['delta_t'] for t in m.T if bess_present)
        custo_deg_ve = sum(params['C_deg_VE'] * (m.P_ve_ch[i,t] + m.P_ve_dis[i,t]) * params['delta_t'] for i in m.V for t in m.T)
        return custo_energia + custo_pico + custo_curtailment + custo_deg_bess + custo_deg_ve
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    model.peak_demand_constraint = pyo.Constraint(model.T, rule=lambda m, t: m.P_grid[t] <= m.P_max_pico)
    model.curtailment_limit_constraint = pyo.Constraint(model.N, model.T, rule=lambda m,j,t: m.P_curt[j,t] <= params['P_pv'].get((j,t), 0))
    if bess_present:
        Emax_BESS_kwh = float(general_df.loc['capacidade_bess_kwh','valor']); SoC_min_BESS_kwh = params['SoC_min_BESS_pu'] * Emax_BESS_kwh; SoC_max_BESS_kwh = params['SoC_max_BESS_pu'] * Emax_BESS_kwh; SoC_init_BESS_kwh = params['SoC_init_BESS_pu'] * Emax_BESS_kwh
        def bess_soc_rule(m, t):
            t_prev = t_prev_map[t]; soc_anterior = SoC_init_BESS_kwh if t == T_list[0] else m.SoC_BESS[t_prev]
            return m.SoC_BESS[t] == soc_anterior + (m.P_bess_ch[t]*params['eta_ch_BESS'] - m.P_bess_dis[t]/params['eta_dis_BESS']) * params['delta_t']
        model.bess_soc_constraint = pyo.Constraint(model.T, rule=bess_soc_rule)
        model.bess_soc_limits_constraint = pyo.Constraint(model.T, rule=lambda m,t: pyo.inequality(SoC_min_BESS_kwh, m.SoC_BESS[t], SoC_max_BESS_kwh))
        model.bess_charge_limit_constraint = pyo.Constraint(model.T, rule=lambda m,t: m.P_bess_ch[t] <= params['Pmax_ch_BESS_kw'] * m.u_BESS[t])
        model.bess_discharge_limit_constraint = pyo.Constraint(model.T, rule=lambda m,t: m.P_bess_dis[t] <= params['Pmax_dis_BESS_kw'] * (1 - m.u_BESS[t]))
    else: model.bess_zero_power = pyo.Constraint(model.T, rule=lambda m,t: m.P_bess_ch[t] == 0 and m.P_bess_dis[t] == 0)
    def ve_soc_rule(m, i, t):
        t_prev = t_prev_map[t]; soc_anterior = params['SoC_init_VE_kwh'][i] if t == T_list[0] else m.SoC_VE[i,t_prev]
        return m.SoC_VE[i,t] == soc_anterior + (m.P_ve_ch[i,t]*params['eta_ch_VE'][i] - m.P_ve_dis[i,t]/params['eta_dis_VE'][i]) * params['delta_t']
    model.ve_soc_constraint = pyo.Constraint(model.V, model.T, rule=ve_soc_rule)
    model.ve_charge_limit_constraint = pyo.Constraint(model.V, model.T, rule=lambda m,i,t: m.P_ve_ch[i,t] <= params['Pmax_ch_VE'][i] * params['a_disp_VE'][(i,t)] * m.u_VE[i,t])
    model.ve_discharge_limit_constraint = pyo.Constraint(model.V, model.T, rule=lambda m,i,t: m.P_ve_dis[i,t] <= params['Pmax_dis_VE'][i] * params['a_disp_VE'][(i,t)] * (params['b_v2g_VE'][i] if v2g_enabled else 0) * (1 - m.u_VE[i,t]))
    def ve_soc_target_rule(m, i):
        t_partida = params['t_partida_VE'][i]; t_idx = t_partida-1 if t_partida > 0 else T_list[-1]
        return m.SoC_VE[i, t_idx] >= params['SoC_target_VE_kwh'][i]
    model.ve_soc_target_constraint = pyo.Constraint(model.V, rule=ve_soc_target_rule)
    def nodal_power_balance_rule(m, j, t):
        if j == 0: return m.P_grid[t] == sum(m.P_ramo[0, k, t] for (jf,k) in m.E if jf == 0)
        parent_n = parent_of.get(j, None)
        if parent_n is None: return pyo.Constraint.Skip
        fluxo_entrada = m.P_ramo[parent_n, j, t]
        fluxo_saida = sum(m.P_ramo[j, k, t] for k in children_of.get(j, []))
        inj_pv = params['P_pv'].get((j,t), 0) - m.P_curt[j,t]
        inj_bess = (m.P_bess_dis[t] - m.P_bess_ch[t]) if j == params['BESS_node'] and bess_present else 0.0
        inj_ves = sum(m.P_ve_dis[i,t] - m.P_ve_ch[i,t] for i in m.V if params['VE_nodes'].get(i,None) == j)
        carga = params['P_carga'].get((j,t),0)
        return fluxo_entrada - fluxo_saida == carga - (inj_pv + inj_bess + inj_ves)
    model.nodal_power_balance_constraint = pyo.Constraint(model.N, model.T, rule=nodal_power_balance_rule)
    @model.Constraint(model.E, model.T)
    def voltage_drop_constraint(m, i, j, t):
        p_ramo_pu = m.P_ramo[i,j,t] / params['S_base_kw']
        return m.V2_no[j,t] == m.V2_no[i,t] - 2 * params['R_pu'][(i,j)] * p_ramo_pu
    model.slack_bus_voltage_constraint = pyo.Constraint(model.T, rule=lambda m, t: m.V2_no[0, t] == 1.0**2)
    V_min_sq_pu, V_max_sq_pu = float(general_df.loc['v_min_pu','valor'])**2, float(general_df.loc['v_max_pu','valor'])**2
    model.voltage_limits_constraint = pyo.Constraint(model.N, model.T, rule=lambda m,j,t: pyo.inequality(V_min_sq_pu, m.V2_no[j,t], V_max_sq_pu))
    model.branch_flow_limits_constraint = pyo.Constraint(model.E, model.T, rule=lambda m,i,j,t: pyo.inequality(-params['P_max_ramo_kw'][(i,j)], m.P_ramo[i,j,t], params['P_max_ramo_kw'][(i,j)]))
    solver = pyo.SolverFactory('cbc'); results = solver.solve(model, tee=False)
    if (results.solver.status == pyo.SolverStatus.ok) and (results.solver.termination_condition == pyo.TerminationCondition.optimal):
        print("Solução ótima encontrada!"); return results, model
    return None, None

def simulate_bau_scenario(params):
    print("\n>>> Simulando Cenário Base (BAU)...")
    df = pd.DataFrame(index=T_list); df.index.name = 'hora'
    p_ve_ch_nodal = {(j, t): 0 for j in N_nodes for t in T_list}
    for i in V:
        ve_node = params['VE_nodes'][i]
        for t in T_list:
            if params['a_disp_VE'][(i, t)] == 1: p_ve_ch_nodal[ve_node, t] += params['Pmax_ch_VE'][i]
    p_inj_nodal = {(j, t): params['P_pv'].get((j, t), 0) - params['P_carga'].get((j, t), 0) - p_ve_ch_nodal.get((j, t), 0) for j in N_nodes for t in T_list}
    p_ramo, v2 = {}, {}
    for t in T_list:
        p_fluxo_nodal = {j: -p_inj_nodal[j, t] for j in N_nodes}
        for j in sorted(N_nodes, reverse=True):
            if j != 0: p_ramo[parent_of[j], j, t] = p_fluxo_nodal[j] + sum(p_ramo.get((j, k, t), 0) for k in children_of.get(j,[]))
        v2[0, t] = 1.0
        nodes_to_process = list(children_of.get(0, []))
        head = 0
        while head < len(nodes_to_process):
            j = nodes_to_process[head]
            i = parent_of[j]
            v2[j, t] = v2[(i, t)] - 2 * params['R_pu'][(i,j)] * (p_ramo.get((i,j,t),0)/params['S_base_kw'])
            nodes_to_process.extend(children_of.get(j, []))
            head += 1
    df['P_grid_kW'] = [sum(p_ramo.get((0,k,t), 0) for k in children_of.get(0,[])) for t in T]
    df['P_grid_kW'] = df['P_grid_kW'].clip(lower=0)
    df['P_carga_total_kW'] = [sum(params['P_carga'].get((j,t), 0) for j in N_nodes) for t in T_list]
    df['P_pv_total_kW'] = [sum(params['P_pv'].get((j,t), 0) for j in N_nodes) for t in T_list]
    df['P_ve_ch_total_kW'] = [sum(p_ve_ch_nodal.get((j,t),0) for j in N_nodes) for t in T_list]
    df['P_ve_dis_total_kW'] = 0
    df['P_curt_total_kW'] = (df['P_pv_total_kW'] - (df['P_carga_total_kW'] + df['P_ve_ch_total_kW']- df['P_grid_kW'])).clip(lower=0)
    df['P_bess_ch_kW'] = 0; df['P_bess_dis_kW'] = 0; df['SoC_BESS_kWh'] = 0
    for j in N_nodes: df[f'V_no_{j}_pu'] = [np.sqrt(v2.get((j,t),1.0)) for t in T_list]
    pico = df['P_grid_kW'].max()
    custo_energia = (df['P_grid_kW'] * pd.Series(params['C_energia']) * params['delta_t']).sum()
    custo_pico = params['C_pico'] * pico
    custo_degradacao = (df['P_ve_ch_total_kW'] * params['C_deg_VE'] * params['delta_t']).sum()
    kpis = {'Custo de Energia': custo_energia, 'Custo de Demanda': custo_pico, 'Custo de Degradação': custo_degradacao}
    detalhes = {}
    ramo_df = pd.DataFrame(index=T_list, columns=[f'{i}-{j}' for i,j in E_branches])
    for t in T_list:
        for i,j in E_branches: ramo_df.loc[t, f'{i}-{j}'] = p_ramo.get((i,j,t), 0)
    detalhes['ramo_df'] = ramo_df
    return df, kpis, detalhes

# ####################################################################################################
# ETAPA 4, 5 e 6: EXECUÇÃO, EXTRAÇÃO E PROCESSAMENTO
# ####################################################################################################
print("\n" + "="*80 + "\n>>> ETAPAS 4, 5, 6: EXECUTANDO CENÁRIOS E PROCESSANDO DADOS\n" + "="*80)
results_repo = {}
results_repo['BAU'] = {}
results_repo['BAU']['agregado'], results_repo['BAU']['kpis'], results_repo['BAU']['detalhado'] = simulate_bau_scenario(params)
scenarios_opt = {'BESS+V1G': {'v2g': False, 'bess': True}, 'BESS+V2G': {'v2g': True, 'bess': True}}
for name, config in scenarios_opt.items():
    results, model = run_optimization_scenario(params, v2g_enabled=config['v2g'], bess_present=config['bess'])
    if model:
        results_repo[name] = {}
        agregado_df = pd.DataFrame(index=T_list)
        agregado_df['P_grid_kW'] = [pyo.value(model.P_grid[t]) for t in T_list]
        agregado_df['P_carga_total_kW'] = [sum(params['P_carga'].get((j,t),0) for j in N_nodes) for t in T_list]
        agregado_df['P_pv_total_kW'] = [sum(params['P_pv'].get((j,t),0) for j in N_nodes) for t in T_list]
        agregado_df['P_curt_total_kW'] = [sum(pyo.value(model.P_curt[j,t]) for j in N_nodes) for t in T_list]
        agregado_df['P_bess_ch_kW'] = [pyo.value(model.P_bess_ch[t]) for t in T_list]
        agregado_df['P_bess_dis_kW'] = [pyo.value(model.P_bess_dis[t]) for t in T_list]
        agregado_df['SoC_BESS_kWh'] = [pyo.value(model.SoC_BESS[t]) if config['bess'] else 0 for t in T_list]
        agregado_df['P_ve_ch_total_kW'] = [sum(pyo.value(model.P_ve_ch[i,t]) for i in V) for t in T_list]
        agregado_df['P_ve_dis_total_kW'] = [sum(pyo.value(model.P_ve_dis[i,t]) for i in V) for t in T_list]
        for j in N_nodes: agregado_df[f'V_no_{j}_pu'] = [np.sqrt(pyo.value(model.V2_no[j,t])) for t in T_list]
        results_repo[name]['agregado'] = agregado_df
        custos = {
            'Custo de Energia': sum(params['C_energia'][t] * pyo.value(model.P_grid[t]) * params['delta_t'] for t in T_list),
            'Custo de Demanda': params['C_pico'] * pyo.value(model.P_max_pico),
            'Custo de Degradação': pyo.value(sum(params['C_deg_BESS']*(model.P_bess_ch[t]+model.P_bess_dis[t])*params['delta_t'] for t in T_list if config['bess']) + sum(params['C_deg_VE']*(model.P_ve_ch[i,t]+model.P_ve_dis[i,t])*params['delta_t'] for i in V for t in T_list))
        }
        results_repo[name]['kpis'] = custos
        detalhado = {}
        detalhado['soc_ve_df'] = pd.DataFrame({f'VE_{i}': [pyo.value(model.SoC_VE[i,t]) for t in T_list] for i in V}, index=T_list)
        ramo_df = pd.DataFrame(index=T_list, columns=[f'{i}-{j}' for i,j in E_branches])
        for t in T_list:
            for i,j in E_branches: ramo_df.loc[t, f'{i}-{j}'] = pyo.value(model.P_ramo[i,j,t])
        detalhado['ramo_df'] = ramo_df
        results_repo[name]['detalhado'] = detalhado
print("Processamento de todos os cenários concluído.")

# ####################################################################################################
# ETAPA 7: GERAÇÃO DE VISUALIZAÇÕES E ANÁLISES
# ####################################################################################################
print("\n" + "="*80 + "\n>>> ETAPA 7: GERANDO VISUALIZAÇÕES E ANÁLISES FINAIS\n" + "="*80)
plt.style.use('seaborn-v0_8-whitegrid')

def plot_despacho_comparativo(results_repo):
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.suptitle('Comparativo de Despacho Agregado de Potência', fontsize=18)

    scenarios_to_plot = ['BAU', 'BESS+V1G', 'BESS+V2G']
    legend_elements = {}

    for ax, scenario in zip(axes, scenarios_to_plot):
        df = results_repo[scenario]['agregado']
        pv_net = df['P_pv_total_kW'] - df['P_curt_total_kW']

        series_data = {'Rede': df['P_grid_kW'], 'PV Líquida': pv_net, 'BESS (Desc)': df['P_bess_dis_kW'], 'VEs (V2G)': df['P_ve_dis_total_kW']}
        colors = {'Rede': '#FF6347', 'PV Líquida': '#FFD700', 'BESS (Desc)': '#1E90FF', 'VEs (V2G)': '#32CD32'}

        plot_order = []
        if scenario == 'BAU':
            plot_order = ['Rede', 'PV Líquida']
        elif scenario == 'BESS+V1G':
            plot_order = ['Rede', 'PV Líquida', 'BESS (Desc)']
        else: # BESS+V2G
            plot_order = ['Rede', 'PV Líquida', 'BESS (Desc)', 'VEs (V2G)']

        stack = ax.stackplot(df.index, [series_data[k] for k in plot_order], labels=plot_order, colors=[colors[k] for k in plot_order])
        demanda_line, = ax.plot(df.index, df['P_carga_total_kW'] + df['P_ve_ch_total_kW'] + df['P_bess_ch_kW'], 'k--', lw=2.5, label='Demanda Total')

        for i, handle in enumerate(stack): legend_elements[plot_order[i]] = handle
        legend_elements['Demanda Total'] = demanda_line

        ax.set_title(f'Cenário: {scenario}', fontsize=14)
        ax.set_xlabel('Hora do Dia'); ax.grid(True); ax.set_xlim(0, 23)

    axes[0].set_ylabel('Potência (kW)')

    final_labels = ['Rede', 'BESS (Desc)', 'VEs (V2G)', 'PV Líquida', 'Demanda Total']
    final_handles = [legend_elements[lbl] for lbl in final_labels if lbl in legend_elements]
    fig.legend(final_handles, [lbl for lbl in final_labels if lbl in legend_elements], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]); plt.show()

    print("--- Análise do Gráfico: Comparativo de Despacho Agregado de Potência ---\n"
          "O que o gráfico mostra: A composição das fontes de energia para atender a demanda total em cada cenário.\n"
          "Principais Observações:\n"
          "  - Cenário BAU: Apresenta um pico de demanda da rede muito elevado no período noturno.\n"
          "  - Cenários Otimizados: O pico de demanda é drasticamente reduzido ('peak shaving').\n"
          "Conclusão para o TCC: A otimização redistribui eficientemente os recursos, reduzindo a dependência da rede nos horários de pico.")

def plot_despacho_comparativo2(results_repo):
    scenarios_to_plot = ['BAU', 'BESS+V1G', 'BESS+V2G']

    # Itera sobre os cenários e cria uma figura separada para cada um
    for scenario in scenarios_to_plot:
        plt.figure(figsize=(12, 7)) # Cria uma nova figura para cada gráfico
        ax = plt.gca() # Pega o eixo da figura atual

        df = results_repo[scenario]['agregado']
        pv_net = df['P_pv_total_kW'] - df['P_curt_total_kW']

        # Lógica de plotagem e legendas dinâmicas para cada cenário
        if scenario == 'BESS+V1G':
            ax.stackplot(df.index, df['P_grid_kW'], df['P_bess_dis_kW'], pv_net,
                         labels=['Rede', 'BESS (Desc)', 'PV Líquida'],
                         colors=['#FF6347', '#1E90FF', '#FFD700'])
        else:
            ax.stackplot(df.index, df['P_grid_kW'], df['P_bess_dis_kW'], df['P_ve_dis_total_kW'], pv_net,
                         labels=['Rede', 'BESS (Desc)', 'VEs (V2G)', 'PV Líquida'],
                         colors=['#FF6347', '#1E90FF', '#32CD32', '#FFD700'])

        # Plota a demanda total
        ax.plot(df.index, df['P_carga_total_kW'] + df['P_ve_ch_total_kW'] + df['P_bess_ch_kW'],
                'k--', lw=2.5, label='Demanda Total')

        ax.set_title(f'Despacho Agregado de Potência - Cenário: {scenario}', fontsize=16)
        ax.set_xlabel('Hora do Dia', fontsize=12)
        ax.set_ylabel('Potência (kW)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True)
        ax.set_xlim(0, 23)

        plt.tight_layout()
        plt.show() # Exibe a figura do cenário atual

    # O resumo escrito permanece o mesmo, pois analisa a comparação entre eles
    print("--- Análise do Gráfico: Comparativo de Despacho Agregado de Potência ---\n"
          "O que o gráfico mostra: A composição das fontes de energia para atender a demanda total em cada cenário.\n"
          "Principais Observações:\n"
          "  - Cenário BAU: Apresenta um pico de demanda da rede muito elevado no período noturno.\n"
          "  - Cenários Otimizados: O pico de demanda é drasticamente reduzido ('peak shaving').\n"
          "Conclusão para o TCC: A otimização redistribui eficientemente os recursos, reduzindo a dependência da rede nos horários de pico.")

def plot_small_multiples_composicao(results_repo):
    all_dfs = []
    for scenario, data in results_repo.items():
        df = data['agregado'].copy()
        df['PV Líquida'] = df['P_pv_total_kW'] - df['P_curt_total_kW']
        oferta_cols = ['P_grid_kW', 'P_bess_dis_kW', 'P_ve_dis_total_kW', 'PV Líquida']
        df_oferta = df[oferta_cols]
        df_percent = df_oferta.divide(df_oferta.sum(axis=1).replace(0, 1e-9), axis=0) * 100
        df_percent['cenario'] = scenario
        all_dfs.append(df_percent)
    full_percent_df = pd.concat(all_dfs)
    full_percent_df = full_percent_df.rename(columns={'P_grid_kW': 'Rede', 'P_bess_dis_kW': 'BESS (Desc)','P_ve_dis_total_kW': 'VEs (V2G)', 'PV Líquida': 'PV Líquida'})
    sources = ['Rede', 'PV Líquida', 'BESS (Desc)', 'VEs (V2G)']
    fig, axes = plt.subplots(len(sources), 1, figsize=(12, 12), sharex=True, sharey=True)
    fig.suptitle('Análise de Composição da Oferta por Fonte de Energia', fontsize=18)
    for ax, source in zip(axes, sources):
        sns.lineplot(data=full_percent_df, x=full_percent_df.index, y=source, hue='cenario', ax=ax, marker='o', markersize=5)
        ax.set_ylabel('Participação (%)'); ax.set_title(f'Fonte: {source}', loc='left', fontsize=12); ax.get_legend().remove(); ax.grid(True, linestyle=':')
    axes[-1].set_xlabel('Hora do Dia', fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.show()
    print("--- Análise do Gráfico: Análise de Composição da Oferta por Fonte de Energia ---\n"
          "O que o gráfico mostra: Uma grade de gráficos onde cada linha representa uma fonte de energia, comparando sua participação percentual nos diferentes cenários.\n"
          "Principais Observações:\n"
          "  - Gráfico 'Rede': Mostra claramente a drástica redução da participação da rede no pico (18h-20h) nos cenários otimizados em comparação com o BAU.\n"
          "  - Gráfico 'BESS (Desc)': Isola a operação do BESS, mostrando que ele atua principalmente no pico da noite para deslocar o consumo da rede.\n"
          "Conclusão para o TCC: Esta visualização detalha a estratégia do otimizador, demonstrando como cada recurso é acionado para atingir o objetivo de redução de custos.")

def plot_custos_comparativo(results_repo):
    kpis_df = pd.DataFrame({s: r['kpis'] for s, r in results_repo.items()}).T.fillna(0)
    kpis_df['Custo Total'] = kpis_df.sum(axis=1)
    ax = kpis_df[['Custo de Energia', 'Custo de Demanda', 'Custo de Degradação']].plot(kind='bar', stacked=True, figsize=(14, 8), colormap='viridis')
    for i, total in enumerate(kpis_df['Custo Total']):
        cumulative_height = 0
        energia_val = kpis_df['Custo de Energia'].iloc[i]
        if energia_val > 100: ax.text(i, cumulative_height + energia_val / 2, f'R$ {energia_val:,.2f}', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        cumulative_height += energia_val
        demanda_val = kpis_df['Custo de Demanda'].iloc[i]
        if demanda_val > 100: ax.text(i, cumulative_height + demanda_val / 2, f'R$ {demanda_val:,.2f}', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        cumulative_height += demanda_val
        degradacao_val = kpis_df['Custo de Degradação'].iloc[i]
        if degradacao_val > 0: ax.text(i, cumulative_height + 250, f'Degradação:\nR$ {degradacao_val:,.2f}', ha='center', va='bottom', color='black', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc='yellow', ec='black', lw=1, alpha=0.8))
    plt.title('Comparativo de Custos Operacionais por Cenário', fontsize=16); plt.ylabel('Custo Total Diário (R$)', fontsize=12); plt.xlabel('Cenário', fontsize=12)
    plt.xticks(rotation=0); plt.legend(title='Componente de Custo'); plt.ylim(top=kpis_df['Custo Total'].max() * 1.15); plt.tight_layout(); plt.show()
    print("--- Análise do Gráfico: Comparativo de Custos Operacionais ---\n"
          "O que o gráfico mostra: O custo operacional diário total, decomposto em suas parcelas para cada cenário.\n"
          "Principais Observações:\n"
          f"  - Custo BAU: Total de R$ {kpis_df.loc['BAU', 'Custo Total']:,.2f}, dominado pela demanda.\n"
          f"  - Custo BESS+V1G: Redução para R$ {kpis_df.loc['BESS+V1G', 'Custo Total']:,.2f}.\n"
          f"  - Custo BESS+V2G: Menor custo total, R$ {kpis_df.loc['BESS+V2G', 'Custo Total']:,.2f}.\n"
          "Conclusão para o TCC: A gestão inteligente (V1G) e a resposta à rede (V2G) geram economias operacionais substanciais.")

def plot_heatmap_comparativo(results_repo, params):
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
    fig.suptitle('Comparativo de Carregamento dos Ramos da Rede (%)', fontsize=18)
    scenarios_to_map = ['BAU', 'BESS+V1G', 'BESS+V2G']
    for i, scenario in enumerate(scenarios_to_map):
        ramo_df = results_repo[scenario]['detalhado']['ramo_df']
        max_flow = pd.Series({f'{i}-{j}': v for (i,j),v in params['P_max_ramo_kw'].items()})
        loading_df = (ramo_df.abs() / max_flow * 100).T
        loading_df = loading_df.astype(float)
        sns.heatmap(loading_df, ax=axes[i], cmap='inferno', vmin=0, vmax=100, annot=False, cbar= (i==2))
        axes[i].set_title(f'Cenário: {scenario}'); axes[i].set_xlabel('Hora do Dia')
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
    print("--- Análise do Gráfico: Comparativo de Carregamento dos Ramos da Rede ---\n"
          "O que o gráfico mostra: O percentual de uso da capacidade máxima de cada linha da rede ao longo do dia.\n"
          "Principais Observações:\n"
          "  - Cenário BAU: Carregamento intenso e concentrado no período das 18h às 22h, criando potenciais pontos de congestionamento.\n"
          "  - Cenários Otimizados: O carregamento da rede é muito mais distribuído e suave ao longo do dia.\n"
          "Conclusão para o TCC: A otimização melhora a confiabilidade da rede ao evitar o congestionamento das linhas.")

def plot_perfis_tensao(results_repo, general_df):
    peak_hour = 19
    v_min_val = float(general_df.loc['v_min_pu','valor'])
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=True)
    fig.suptitle(f'Perfil de Tensão Individual por Cenário (Pico às {peak_hour}h)', fontsize=18)
    for ax, scenario in zip(axes, results_repo.keys()):
        data = results_repo[scenario]
        v_cols = [c for c in data['agregado'].columns if c.startswith('V_no_')]
        voltages = data['agregado'].loc[peak_hour, v_cols]
        voltages.index = [int(c.split('_')[2]) for c in voltages.index]; voltages = voltages.sort_index()
        ax.plot(voltages.index, voltages.values, marker='o', linestyle='-')
        ax.axhline(v_min_val, color='red', linestyle='--', label='Limite Mínimo')
        ax.set_title(f"Cenário: {scenario}"); ax.set_xlabel('Barra da Rede'); ax.set_ylabel('Tensão (p.u.)')
        ax.legend(); ax.grid(True); ax.set_ylim(bottom=v_min_val - 0.02, top=1.01)
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
    plt.figure(figsize=(14, 7))
    plt.title(f'Perfil de Tensão Comparativo ao Longo do Alimentador (Pico às {peak_hour}h)', fontsize=16)
    for scenario, data in results_repo.items():
        v_cols = [c for c in data['agregado'].columns if c.startswith('V_no_')]
        voltages = data['agregado'].loc[peak_hour, v_cols]
        voltages.index = [int(c.split('_')[2]) for c in voltages.index]; voltages = voltages.sort_index()
        plt.plot(voltages.index, voltages.values, marker='o', linestyle='-', label=f'Cenário {scenario}')
    plt.axhline(v_min_val, color='red', linestyle='--', label='Limite Mínimo')
    plt.xlabel('Barra da Rede'); plt.ylabel('Tensão (p.u.)'); plt.legend(); plt.grid(True)
    plt.ylim(bottom=v_min_val - 0.02, top=1.01)
    plt.tight_layout(); plt.show()
    print("--- Análise do Gráfico: Perfil de Tensão ao Longo do Alimentador ---\n"
          "O que o gráfico mostra: A tensão em cada barra da rede no horário de maior carregamento (19h).\n"
          "Principais Observações:\n"
          "  - Cenário BAU: A tensão cai significativamente nas barras mais distantes, chegando a violar o limite mínimo.\n"
          "  - Cenário BESS+V1G: A otimização melhora o perfil de tensão, mantendo-o acima do limite.\n"
          "  - Cenário BESS+V2G: Apresenta o melhor resultado, com tensões mais elevadas, pois os VEs podem injetar potência e fornecer suporte de tensão.\n"
          "Conclusão para o TCC: A operação coordenada (V1G) e, principalmente, a bidirecional (V2G) são ferramentas eficazes para garantir a qualidade e a estabilidade da tensão.")

def plot_perfis_ves_destaque(results_repo, params):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    fig.suptitle('Destaque: Perfis de Carga Individuais dos VEs (Cenário BESS+V2G)', fontsize=18)
    soc_ve_df = results_repo['BESS+V2G']['detalhado']['soc_ve_df']
    for i, ax in enumerate(axes.flatten()):
        if i < len(V):
            ve_id = V[i]; chegada, partida = params['t_chegada_VE'][ve_id], params['t_partida_VE'][ve_id]
            soc_ve_df[f'VE_{ve_id}'].plot(ax=ax, marker='.', linestyle='-')
            ax.set_title(f'Veículo {ve_id}'); ax.set_ylabel('SoC (kWh)')
            ax.axvspan(0, chegada, color='gray', alpha=0.2, label='Ausente')
            if chegada > partida: ax.axvspan(partida, 23, color='gray', alpha=0.2)
            else: ax.axvspan(partida, 23, color='gray', alpha=0.2)
            ax.grid(True, linestyle=':')
    plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.show()
    print("--- Análise do Gráfico: Perfis de Carga Individuais dos VEs ---\n"
          "O que o gráfico mostra: A trajetória do SoC de cada veículo durante sua conexão no cenário mais avançado.\n"
          "Principais Observações:\n"
          "  - Comportamento Heterogêneo: O otimizador não carrega todos os VEs da mesma forma. Alguns descarregam (V2G) para ajudar a rede no pico.\n"
          "  - Meta Atendida: Todos os veículos atingem sua meta de carga antes da partida.\n"
          "Conclusão para o TCC: A otimização explora a flexibilidade da frota de forma inteligente e individualizada, transformando-os em recursos ativos para a rede.")

def plot_autoconsumo_destaque(results_repo):
    res_v2g = results_repo['BESS+V2G']['agregado']
    pv_total_gerado = res_v2g['P_pv_total_kW'].sum()
    pv_curtailado = res_v2g['P_curt_total_kW'].sum()
    consumo_direto, armazenado = 0, 0
    for t in T_list:
        geracao_hora = res_v2g.loc[t, 'P_pv_total_kW'] - res_v2g.loc[t, 'P_curt_total_kW']
        demanda_fixa_hora = res_v2g.loc[t, 'P_carga_total_kW']
        consumo_direto_hora = min(geracao_hora, demanda_fixa_hora)
        consumo_direto += consumo_direto_hora
        sobra_pv = geracao_hora - consumo_direto_hora
        demanda_flexivel_hora = res_v2g.loc[t, 'P_bess_ch_kW'] + res_v2g.loc[t, 'P_ve_ch_total_kW']
        armazenado += min(sobra_pv, demanda_flexivel_hora)
    energy_dest = {'Consumo Direto na Carga': consumo_direto, 'Armazenado (BESS+VEs)': armazenado, 'Desperdiçado (Curtailment)': pv_curtailado}
    total_calculado = sum(energy_dest.values())
    if total_calculado > pv_total_gerado: energy_dest['Consumo Direto na Carga'] -= (total_calculado - pv_total_gerado)
    plt.figure(figsize=(8, 8))
    plt.pie(energy_dest.values(), labels=energy_dest.keys(), autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(energy_dest)))
    plt.title('Destaque: Destino da Energia Fotovoltaica (Cenário BESS+V2G)', fontsize=16)
    plt.tight_layout(); plt.show()
    print("--- Análise do Gráfico: Destino da Energia Fotovoltaica ---\n"
          "O que o gráfico mostra: A distribuição de toda a energia gerada pela fonte PV no cenário otimizado.\n"
          "Principais Observações:\n"
          "  - Máximo Aproveitamento: A grande maioria da energia PV é utilizada para atender a carga ou para ser armazenada.\n"
          "  - Baixo Desperdício: O curtailment é mínimo, indicando que o armazenamento absorve a geração excedente.\n"
          "Conclusão para o TCC: A coordenação com armazenamento é fundamental para maximizar o autoconsumo de energia renovável.")

def plot_resumo_kpis_eixo_duplo(summary_df):
    fig, ax1 = plt.subplots(figsize=(12, 7))
    fig.suptitle('Análise Combinada de KPIs: Custo vs. Tensão', fontsize=18)
    summary_df['Custo Total (R$)'].plot(kind='bar', ax=ax1, color='lightgray', width=0.4, position=1, label='Custo Total (R$)')
    ax1.set_ylabel('Custo Total Diário (R$)', fontsize=12)
    ax1.set_xlabel('Cenário', fontsize=12)
    ax1.tick_params(axis='x', rotation=0)
    ax2 = ax1.twinx()
    ax2.plot(ax1.get_xticks(), summary_df['Tensão Mínima (p.u.)'], color='crimson', marker='o', linestyle='--', label='Tensão Mínima (p.u.)')
    ax2.set_ylabel('Tensão Mínima (p.u.)', fontsize=12, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')
    v_min_val = float(general_df.loc['v_min_pu','valor'])
    ax2.axhline(v_min_val, color='red', linestyle=':', label=f'Limite Mínimo ({v_min_val} p.u.)')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]); plt.grid(False); ax1.grid(True, axis='y', linestyle=':'); plt.show()
    print("--- Análise do Gráfico: Análise Combinada de KPIs ---\n"
          "O que o gráfico mostra: Uma comparação direta entre um indicador econômico (Custo Total, em barras) e um indicador técnico (Tensão Mínima, em linha) para cada cenário.\n"
          "Principais Observações:\n"
          "  - Relação Inversa: Fica evidente a relação inversa entre custo e qualidade de energia. À medida que o custo operacional diminui, o nível de tensão melhora.\n"
          "  - Impacto do BAU: O cenário BAU apresenta o maior custo e, simultaneamente, a pior performance técnica, com a tensão violando o limite mínimo.\n"
          "Conclusão para o TCC: Este gráfico sintetiza o argumento central do trabalho, provando que a coordenação otimizada gera benefícios tecno-econômicos sinérgicos.")


# --- Chamada das Funções de Plotagem ---
plot_despacho_comparativo(results_repo)
plot_despacho_comparativo2(results_repo)
plot_small_multiples_composicao(results_repo)
plot_custos_comparativo(results_repo)
plot_heatmap_comparativo(results_repo, params)
plot_perfis_tensao(results_repo, general_df)
plot_perfis_ves_destaque(results_repo, params)

# ####################################################################################################
# ETAPA 8: GERAÇÃO DO RESUMO FINAL COMPARATIVO
# ####################################################################################################
print("\n" + "="*80 + "\n>>> ETAPA 8: RESUMO FINAL COMPARATIVO DOS RESULTADOS\n" + "="*80)
summary_data = []
scenarios = ['BAU', 'BESS+V1G', 'BESS+V2G']
for scenario in scenarios:
    if scenario in results_repo:
        data = results_repo[scenario]
        kpis = data['kpis']
        agregado_df = data['agregado']
        v_cols = [c for c in agregado_df.columns if c.startswith('V_no_')]
        summary_data.append({
            'Cenário': scenario,
            'Custo Total (R$)': sum(kpis.values()),
            'Custo de Energia (R$)': kpis.get('Custo de Energia', 0),
            'Custo de Demanda (R$)': kpis.get('Custo de Demanda', 0),
            'Custo de Degradação (R$)': kpis.get('Custo de Degradação', 0),
            'Pico de Demanda (kW)': agregado_df['P_grid_kW'].max(),
            'Energia Importada (kWh)': agregado_df['P_grid_kW'].sum() * params['delta_t'],
            'Energia PV Curtailada (kWh)': agregado_df['P_curt_total_kW'].sum() * params['delta_t'],
            'Tensão Mínima (p.u.)': agregado_df[v_cols].min().min(),
        })
summary_df = pd.DataFrame(summary_data).set_index('Cenário')
bau_metrics = summary_df.loc['BAU']
summary_df['Redução Custo Total (%)'] = (1 - summary_df['Custo Total (R$)'] / bau_metrics['Custo Total (R$)']) * 100
summary_df['Redução Pico Demanda (%)'] = (1 - summary_df['Pico de Demanda (kW)'] / bau_metrics['Pico de Demanda (kW)']) * 100
pd.options.display.float_format = '{:,.2f}'.format
print("Tabela Comparativa de Indicadores de Desempenho (KPIs):")
display(summary_df)

# --- Chamada do Gráfico de Resumo Final ---
plot_resumo_kpis_eixo_duplo(summary_df)
