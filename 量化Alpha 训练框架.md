

# 顶级量化基金 Alpha 训练框架深度解析

## 一、整体架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    Alpha Research Pipeline                       │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Universe │→│  Alpha   │→│ Portfolio │→│  Execution    │  │
│  │ & Data   │  │ Engine   │  │ Construct │  │  & Risk Mgmt  │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│       ↑              ↑             ↑              ↑             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Backtesting & Simulation Engine             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、数据层 (Data Infrastructure)

### 2.1 数据分类与来源

```python
class AlphaDataInfrastructure:
    """
    顶级量化基金数据基础设施
    Two Sigma: 强调替代数据 + NLP
    AQR: 强调学术因子 + 宏观数据
    WorldQuant: 强调海量算子搜索 + 全球覆盖
    """

    DATA_TAXONOMY = {
        # ========== 传统市场数据 ==========
        "market_data": {
            "price_volume": {
                "sources": ["NYSE TAQ", "NASDAQ ITCH", "CME", "ICE"],
                "granularity": "tick-level / L2 orderbook",
                "fields": ["OHLCV", "bid/ask", "depth", "trade_condition"],
                "history": "20+ years",
            },
            "fundamental": {
                "sources": ["Compustat", "WorldScope", "Bloomberg", "FactSet"],
                "fields": ["income_stmt", "balance_sheet", "cashflow", "estimates"],
                "frequency": "quarterly / annual",
                "point_in_time": True,  # 关键：避免前视偏差
            },
            "reference": {
                "sources": ["CRSP", "Axioma", "MSCI", "S&P"],
                "fields": ["industry", "market_cap", "index_membership",
                          "corporate_actions", "share_changes"],
            },
        },

        # ========== 替代数据 (Two Sigma 重点) ==========
        "alternative_data": {
            "satellite_imagery": {
                "providers": ["Orbital Insight", "RS Metrics", "Descartes Labs"],
                "use_cases": ["retail_foot_traffic", "oil_storage",
                             "crop_yield", "construction_activity"],
            },
            "web_scraping": {
                "targets": ["job_postings", "product_pricing", "app_downloads",
                           "social_media_sentiment", "SEC_filings"],
            },
            "transaction_data": {
                "providers": ["Second Measure", "Earnest Research", "Yodlee"],
                "use_cases": ["revenue_nowcasting", "market_share_tracking"],
            },
            "nlp_text": {
                "sources": ["earnings_calls", "news_wire", "analyst_reports",
                           "patent_filings", "reddit/twitter"],
            },
            "geolocation": {
                "providers": ["SafeGraph", "Placer.ai"],
                "use_cases": ["store_visits", "supply_chain_monitoring"],
            },
        },
    }
```

### 2.2 Point-in-Time 数据管理

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PointInTimeDatabase:
    """
    PIT数据库 —— 防止前视偏差的核心
    所有顶级基金都使用这一原则
    """

    def __init__(self):
        self.pit_store = {}  # {(entity, field): [(knowledge_date, value, as_of_date)]}

    def insert(self, entity: str, field: str, value: float,
               as_of_date: str, knowledge_date: str):
        """
        as_of_date: 数据所属的时间段（如2023Q3财报）
        knowledge_date: 投资者可以知道这个数据的最早日期（如发布日2023-11-05）
        """
        key = (entity, field)
        if key not in self.pit_store:
            self.pit_store[key] = []
        self.pit_store[key].append({
            'knowledge_date': pd.Timestamp(knowledge_date),
            'value': value,
            'as_of_date': pd.Timestamp(as_of_date),
        })

    def query(self, entity: str, field: str, query_date: str) -> float:
        """
        查询在query_date时投资者能获得的最新数据
        关键：只返回 knowledge_date <= query_date 的记录
        """
        key = (entity, field)
        if key not in self.pit_store:
            return np.nan

        query_ts = pd.Timestamp(query_date)
        available = [
            r for r in self.pit_store[key]
            if r['knowledge_date'] <= query_ts
        ]
        if not available:
            return np.nan

        # 返回 knowledge_date 最近的记录
        latest = max(available, key=lambda x: x['knowledge_date'])
        return latest['value']


class SurvivalBiasFreeUniverse:
    """
    无生存偏差的投资宇宙
    必须包含已退市/破产/被并购的股票
    """

    def __init__(self):
        self.universe_history = {}  # date -> set of tickers

    def get_universe(self, date: str, filters: dict = None) -> list:
        """
        返回在给定日期的实际可交易证券列表
        包含后来退市的股票，排除尚未上市的股票
        """
        dt = pd.Timestamp(date)
        universe = self.universe_history.get(dt, set())

        if filters:
            if 'min_mcap' in filters:
                universe = {s for s in universe
                           if self._get_mcap(s, dt) >= filters['min_mcap']}
            if 'min_adv' in filters:
                universe = {s for s in universe
                           if self._get_adv(s, dt) >= filters['min_adv']}
            if 'min_price' in filters:
                universe = {s for s in universe
                           if self._get_price(s, dt) >= filters['min_price']}

        return sorted(universe)

    def _get_mcap(self, ticker, date):
        pass  # 实际实现查询市值

    def _get_adv(self, ticker, date):
        pass  # 实际实现查询日均成交额

    def _get_price(self, ticker, date):
        pass  # 实际实现查询价格
```

---

## 三、Alpha 因子引擎

### 3.1 WorldQuant 式算子搜索框架 (Alpha Mining)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Callable
import itertools


# ========== 算子定义 ==========

class Operator(ABC):
    """WorldQuant-style alpha算子基类"""

    @abstractmethod
    def __call__(self, *args) -> pd.DataFrame:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class RankOperator(Operator):
    """截面排名"""
    @property
    def name(self):
        return "rank"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, pct=True)


class DeltaOperator(Operator):
    """时序差分"""
    def __init__(self, period: int):
        self.period = period

    @property
    def name(self):
        return f"delta_{self.period}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df - df.shift(self.period)


class TsRankOperator(Operator):
    """时序排名"""
    def __init__(self, window: int):
        self.window = window

    @property
    def name(self):
        return f"ts_rank_{self.window}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1],
            raw=False
        )


class TsStdOperator(Operator):
    """时序标准差"""
    def __init__(self, window: int):
        self.window = window

    @property
    def name(self):
        return f"ts_std_{self.window}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rolling(self.window).std()


class DecayLinearOperator(Operator):
    """线性衰减加权均值"""
    def __init__(self, window: int):
        self.window = window

    @property
    def name(self):
        return f"decay_linear_{self.window}"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        weights = np.arange(1, self.window + 1, dtype=float)
        weights /= weights.sum()
        return df.rolling(self.window).apply(
            lambda x: np.dot(x, weights), raw=True
        )


class IndustryNeutralizeOperator(Operator):
    """行业中性化"""
    def __init__(self, industry_map: dict):
        self.industry_map = industry_map

    @property
    def name(self):
        return "indneutralize"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        for date in df.index:
            row = df.loc[date]
            industries = pd.Series({col: self.industry_map.get(col, 'Unknown')
                                   for col in df.columns})
            for ind in industries.unique():
                mask = industries == ind
                ind_mean = row[mask].mean()
                result.loc[date, mask] -= ind_mean
        return result


class CorrelationOperator(Operator):
    """时序相关性"""
    def __init__(self, window: int):
        self.window = window

    @property
    def name(self):
        return f"correlation_{self.window}"

    def __call__(self, df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        return df1.rolling(self.window).corr(df2)


# ========== 数据字段 ==========

class DataField:
    """原始数据字段"""

    FIELDS = {
        # 价量数据
        'open': 'Open price',
        'high': 'High price',
        'low': 'Low price',
        'close': 'Close price',
        'volume': 'Trading volume',
        'vwap': 'Volume weighted average price',
        'returns': 'Daily returns',
        'adv20': '20-day average daily volume',
        'cap': 'Market capitalization',

        # 基本面数据 (PIT)
        'sales': 'Revenue',
        'earnings': 'Net income',
        'book_value': 'Book value of equity',
        'cashflow': 'Operating cash flow',
        'debt': 'Total debt',
        'assets': 'Total assets',
        'dividend': 'Dividend per share',

        # 分析师数据
        'est_eps': 'Consensus EPS estimate',
        'est_rev': 'Consensus revenue estimate',
        'est_revision': 'Estimate revision',
        'recommendation': 'Analyst recommendation',

        # 替代数据
        'sentiment': 'News sentiment score',
        'short_interest': 'Short interest ratio',
        'insider_buy': 'Insider buying volume',
    }


# ========== Alpha 表达式树 ==========

class AlphaExpression:
    """
    WorldQuant-style Alpha表达式
    用树结构表示 alpha = f(operators, data_fields)
    """

    def __init__(self, expression_str: str):
        self.expression_str = expression_str
        self.tree = self._parse(expression_str)

    def _parse(self, expr_str):
        """解析表达式字符串为计算树"""
        # 简化实现 — 实际使用递归下降解析器
        pass

    def evaluate(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """对数据计算alpha值"""
        pass

    def __repr__(self):
        return f"Alpha({self.expression_str})"


# ========== 示例 WorldQuant Alpha ==========

WORLDQUANT_ALPHA_EXAMPLES = {
    # Alpha#1: 经典反转
    "alpha_001": "rank(ts_argmax(power(((returns < 0) ? ts_std(returns, 20) : close), 2), 5)) - 0.5",

    # Alpha#6: 开盘-成交量相关性
    "alpha_006": "-1 * correlation(open, volume, 10)",

    # Alpha#12: 符号 × 成交量变化
    "alpha_012": "sign(delta(volume, 1)) * (-1 * delta(close, 1))",

    # Alpha#24: 价格偏离 SMA
    "alpha_024": "-1 * ((delta(close - ts_mean(close, 100), 100) / delay(close, 100)) < 0.05 ? "
                 "delta(close, 3) : -1 * (close - ts_min(close, 100)))",

    # Alpha#41: VWAP改进
    "alpha_041": "power(high * low, 0.5) - vwap",

    # Alpha#101: 价量关系
    "alpha_101": "(close - open) / ((high - low) + 0.001)",
}
```

### 3.2 Alpha 遗传搜索引擎 (Genetic Programming)

```python
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GPNode:
    """遗传编程树节点"""
    node_type: str  # 'operator', 'data', 'constant'
    value: str      # 操作符名称/数据字段名/常数值
    children: list = field(default_factory=list)
    arity: int = 0  # 操作符需要的参数数量

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def to_expression(self) -> str:
        if self.node_type == 'data':
            return self.value
        elif self.node_type == 'constant':
            return str(self.value)
        elif self.arity == 1:
            return f"{self.value}({self.children[0].to_expression()})"
        elif self.arity == 2:
            child_exprs = [c.to_expression() for c in self.children]
            if self.value in ['+', '-', '*', '/']:
                return f"({child_exprs[0]} {self.value} {child_exprs[1]})"
            else:
                return f"{self.value}({', '.join(child_exprs)})"
        else:
            child_exprs = [c.to_expression() for c in self.children]
            return f"{self.value}({', '.join(child_exprs)})"


class AlphaGeneticProgramming:
    """
    WorldQuant / Two Sigma 式遗传编程 Alpha搜索

    核心思想:
    1. 随机生成alpha表达式树种群
    2. 通过适应度函数(IC/IR/Sharpe)评估
    3. 选择、交叉、变异产生下一代
    4. 迭代搜索直到找到有效alpha
    """

    # 定义搜索空间
    UNARY_OPS = [
        ('rank', 1), ('abs', 1), ('log', 1), ('sign', 1),
        ('ts_rank_5', 1), ('ts_rank_10', 1), ('ts_rank_20', 1),
        ('ts_mean_5', 1), ('ts_mean_10', 1), ('ts_mean_20', 1),
        ('ts_std_5', 1), ('ts_std_10', 1), ('ts_std_20', 1),
        ('delta_1', 1), ('delta_5', 1), ('delta_10', 1),
        ('delay_1', 1), ('delay_5', 1),
        ('decay_linear_5', 1), ('decay_linear_10', 1),
        ('ts_min_5', 1), ('ts_max_5', 1),
        ('ts_min_10', 1), ('ts_max_10', 1),
        ('indneutralize', 1),
    ]

    BINARY_OPS = [
        ('+', 2), ('-', 2), ('*', 2), ('/', 2),
        ('correlation_5', 2), ('correlation_10', 2),
        ('covariance_5', 2), ('covariance_10', 2),
        ('ts_regression_residual_10', 2),
    ]

    DATA_FIELDS = [
        'open', 'high', 'low', 'close', 'volume', 'vwap',
        'returns', 'adv20', 'cap',
    ]

    def __init__(self, population_size=1000, max_depth=6,
                 tournament_size=7, crossover_prob=0.7,
                 mutation_prob=0.2, elitism_ratio=0.05):
        self.population_size = population_size
        self.max_depth = max_depth
        self.tournament_size = tournament_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elitism_ratio = elitism_ratio
        self.population = []
        self.best_alphas = []

    def _random_terminal(self) -> GPNode:
        """生成随机终端节点"""
        if random.random() < 0.8:
            field = random.choice(self.DATA_FIELDS)
            return GPNode(node_type='data', value=field, arity=0)
        else:
            const = round(random.uniform(-1, 1), 2)
            return GPNode(node_type='constant', value=const, arity=0)

    def _random_operator(self) -> Tuple[str, int]:
        """随机选择操作符"""
        all_ops = self.UNARY_OPS + self.BINARY_OPS
        return random.choice(all_ops)

    def _generate_tree(self, max_depth: int, method='grow') -> GPNode:
        """
        生成随机表达式树
        method: 'grow' (变深度) / 'full' (满树)
        """
        if max_depth <= 1:
            return self._random_terminal()

        if method == 'grow' and random.random() < 0.3:
            return self._random_terminal()

        op_name, arity = self._random_operator()
        node = GPNode(
            node_type='operator',
            value=op_name,
            arity=arity,
            children=[self._generate_tree(max_depth - 1, method) for _ in range(arity)]
        )
        return node

    def initialize_population(self):
        """Ramped half-and-half 初始化"""
        self.population = []
        for i in range(self.population_size):
            depth = random.randint(2, self.max_depth)
            method = 'grow' if i % 2 == 0 else 'full'
            tree = self._generate_tree(depth, method)
            self.population.append(tree)

    def fitness(self, tree: GPNode, data: dict,
                forward_returns: pd.DataFrame) -> dict:
        """
        适应度评估 —— 多维度评分

        核心指标:
        - IC (Information Coefficient): rank correlation with forward returns
        - IR (Information Ratio): IC_mean / IC_std
        - Turnover: 换手率
        - Drawdown: 最大回撤
        - Fitness = IR * sqrt(|avg_IC|) * decay(turnover)
        """
        try:
            alpha_values = self._evaluate_tree(tree, data)

            if alpha_values is None or alpha_values.isna().all().all():
                return {'fitness': -np.inf, 'IC': 0, 'IR': 0, 'turnover': 1}

            # 截面排名标准化
            alpha_ranked = alpha_values.rank(axis=1, pct=True) - 0.5

            # 计算每日IC (Spearman rank correlation)
            daily_ic = []
            for date in alpha_ranked.index:
                if date in forward_returns.index:
                    mask = alpha_ranked.loc[date].notna() & forward_returns.loc[date].notna()
                    if mask.sum() > 30:
                        ic = alpha_ranked.loc[date][mask].corr(
                            forward_returns.loc[date][mask], method='spearman'
                        )
                        daily_ic.append(ic)

            if len(daily_ic) < 60:
                return {'fitness': -np.inf, 'IC': 0, 'IR': 0, 'turnover': 1}

            ic_series = pd.Series(daily_ic)
            avg_ic = ic_series.mean()
            ic_std = ic_series.std()
            ir = avg_ic / ic_std if ic_std > 0 else 0

            # 换手率
            turnover = (alpha_ranked.diff().abs().sum(axis=1) /
                       alpha_ranked.abs().sum(axis=1)).mean()

            # 复合适应度
            turnover_penalty = np.exp(-2 * max(turnover - 0.3, 0))
            fitness_score = ir * np.sqrt(abs(avg_ic)) * turnover_penalty

            # 复杂度惩罚
            complexity_penalty = 1.0 / (1.0 + 0.01 * tree.size())
            fitness_score *= complexity_penalty

            return {
                'fitness': fitness_score,
                'IC': avg_ic,
                'IR': ir,
                'turnover': turnover,
                'ic_series': ic_series,
            }

        except Exception:
            return {'fitness': -np.inf, 'IC': 0, 'IR': 0, 'turnover': 1}

    def _evaluate_tree(self, tree: GPNode, data: dict) -> Optional[pd.DataFrame]:
        """递归计算表达式树"""
        if tree.node_type == 'data':
            return data.get(tree.value)
        elif tree.node_type == 'constant':
            ref_df = data.get('close')
            return pd.DataFrame(
                tree.value, index=ref_df.index, columns=ref_df.columns
            )
        else:
            child_values = [self._evaluate_tree(c, data) for c in tree.children]
            if any(v is None for v in child_values):
                return None
            return self._apply_operator(tree.value, child_values)

    def _apply_operator(self, op_name: str,
                        args: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """应用操作符"""
        try:
            if op_name == '+':
                return args[0] + args[1]
            elif op_name == '-':
                return args[0] - args[1]
            elif op_name == '*':
                return args[0] * args[1]
            elif op_name == '/':
                return args[0] / args[1].replace(0, np.nan)
            elif op_name == 'rank':
                return args[0].rank(axis=1, pct=True)
            elif op_name == 'abs':
                return args[0].abs()
            elif op_name == 'log':
                return np.log(args[0].clip(lower=1e-10))
            elif op_name == 'sign':
                return np.sign(args[0])
            elif op_name.startswith('ts_mean_'):
                w = int(op_name.split('_')[-1])
                return args[0].rolling(w).mean()
            elif op_name.startswith('ts_std_'):
                w = int(op_name.split('_')[-1])
                return args[0].rolling(w).std()
            elif op_name.startswith('ts_rank_'):
                w = int(op_name.split('_')[-1])
                return args[0].rolling(w).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
            elif op_name.startswith('delta_'):
                d = int(op_name.split('_')[-1])
                return args[0] - args[0].shift(d)
            elif op_name.startswith('delay_'):
                d = int(op_name.split('_')[-1])
                return args[0].shift(d)
            elif op_name.startswith('decay_linear_'):
                w = int(op_name.split('_')[-1])
                weights = np.arange(1, w + 1, dtype=float)
                weights /= weights.sum()
                return args[0].rolling(w).apply(lambda x: np.dot(x, weights), raw=True)
            elif op_name.startswith('correlation_'):
                w = int(op_name.split('_')[-1])
                return args[0].rolling(w).corr(args[1])
            elif op_name.startswith('ts_min_'):
                w = int(op_name.split('_')[-1])
                return args[0].rolling(w).min()
            elif op_name.startswith('ts_max_'):
                w = int(op_name.split('_')[-1])
                return args[0].rolling(w).max()
            elif op_name == 'indneutralize':
                return args[0].sub(args[0].mean(axis=1), axis=0)  # 简化版
            else:
                return None
        except Exception:
            return None

    def tournament_selection(self, fitnesses: list) -> GPNode:
        """锦标赛选择"""
        candidates = random.sample(
            list(zip(self.population, fitnesses)),
            self.tournament_size
        )
        winner = max(candidates, key=lambda x: x[1]['fitness'])
        return deepcopy(winner[0])

    def crossover(self, parent1: GPNode, parent2: GPNode) -> Tuple[GPNode, GPNode]:
        """子树交叉"""
        child1 = deepcopy(parent1)
        child2 = deepcopy(parent2)

        # 随机选择交叉点
        nodes1 = self._get_all_nodes(child1)
        nodes2 = self._get_all_nodes(child2)

        if len(nodes1) < 2 or len(nodes2) < 2:
            return child1, child2

        # 选择交叉点 (倾向于选择函数节点)
        point1 = random.choice(nodes1[1:])
        point2 = random.choice(nodes2[1:])

        # 交换子树 (通过引用父节点)
        self._swap_subtrees(child1, point1, point2)
        self._swap_subtrees(child2, point2, point1)

        # 深度限制
        if child1.depth() > self.max_depth:
            child1 = parent1
        if child2.depth() > self.max_depth:
            child2 = parent2

        return child1, child2

    def mutate(self, tree: GPNode) -> GPNode:
        """变异操作"""
        mutant = deepcopy(tree)
        mutation_type = random.choice(['subtree', 'point', 'hoist', 'shrink'])

        if mutation_type == 'subtree':
            # 替换随机子树
            nodes = self._get_all_nodes(mutant)
            if nodes:
                target = random.choice(nodes)
                max_d = max(1, self.max_depth - self._node_depth(mutant, target))
                new_subtree = self._generate_tree(max_d, 'grow')
                self._replace_node(mutant, target, new_subtree)

        elif mutation_type == 'point':
            # 替换单个节点（保持arity不变）
            nodes = self._get_all_nodes(mutant)
            target = random.choice(nodes)
            if target.node_type == 'data':
                target.value = random.choice(self.DATA_FIELDS)
            elif target.node_type == 'operator':
                same_arity_ops = [
                    (n, a) for n, a in (self.UNARY_OPS + self.BINARY_OPS)
                    if a == target.arity
                ]
                if same_arity_ops:
                    new_op, _ = random.choice(same_arity_ops)
                    target.value = new_op

        elif mutation_type == 'shrink':
            # 将随机子树替换为终端节点
            nodes = [n for n in self._get_all_nodes(mutant)
                    if n.node_type == 'operator']
            if nodes:
                target = random.choice(nodes)
                terminal = self._random_terminal()
                self._replace_node(mutant, target, terminal)

        return mutant

    def evolve(self, data: dict, forward_returns: pd.DataFrame,
               n_generations: int = 100) -> List[dict]:
        """
        主进化循环
        """
        self.initialize_population()
        all_results = []

        for gen in range(n_generations):
            # 评估适应度
            fitnesses = [
                self.fitness(tree, data, forward_returns)
                for tree in self.population
            ]

            # 记录最佳
            best_idx = max(range(len(fitnesses)),
                          key=lambda i: fitnesses[i]['fitness'])
            best_fitness = fitnesses[best_idx]
            best_tree = self.population[best_idx]

            print(f"Gen {gen:3d} | Best Fitness: {best_fitness['fitness']:.4f} | "
                  f"IC: {best_fitness['IC']:.4f} | IR: {best_fitness['IR']:.4f} | "
                  f"Expr: {best_tree.to_expression()[:80]}")

            if best_fitness['fitness'] > 0.05:  # 有意义的阈值
                all_results.append({
                    'generation': gen,
                    'expression': best_tree.to_expression(),
                    'tree': deepcopy(best_tree),
                    **best_fitness,
                })

            # 精英保留
            n_elite = int(self.population_size * self.elitism_ratio)
            elite_indices = sorted(
                range(len(fitnesses)),
                key=lambda i: fitnesses[i]['fitness'],
                reverse=True
            )[:n_elite]
            new_population = [deepcopy(self.population[i]) for i in elite_indices]

            # 生成下一代
            while len(new_population) < self.population_size:
                if random.random() < self.crossover_prob:
                    p1 = self.tournament_selection(fitnesses)
                    p2 = self.tournament_selection(fitnesses)
                    c1, c2 = self.crossover(p1, p2)
                    new_population.extend([c1, c2])
                else:
                    p = self.tournament_selection(fitnesses)
                    if random.random() < self.mutation_prob:
                        p = self.mutate(p)
                    new_population.append(p)

            self.population = new_population[:self.population_size]

        return all_results

    def _get_all_nodes(self, tree: GPNode) -> list:
        """获取树的所有节点"""
        nodes = [tree]
        for child in tree.children:
            nodes.extend(self._get_all_nodes(child))
        return nodes

    def _node_depth(self, root, target, current_depth=0):
        if root is target:
            return current_depth
        for child in root.children:
            d = self._node_depth(child, target, current_depth + 1)
            if d >= 0:
                return d
        return -1

    def _replace_node(self, root, target, replacement):
        for i, child in enumerate(root.children):
            if child is target:
                root.children[i] = replacement
                return True
            if self._replace_node(child, target, replacement):
                return True
        return False

    def _swap_subtrees(self, root, old_node, new_node):
        self._replace_node(root, old_node, deepcopy(new_node))
```

### 3.3 AQR 式学术因子框架

```python
from enum import Enum


class FactorCategory(Enum):
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_RISK = "low_risk"
    SIZE = "size"
    CARRY = "carry"
    LIQUIDITY = "liquidity"
    SENTIMENT = "sentiment"


class AQRStyleFactorLibrary:
    """
    AQR 式学术因子库
    基于 Fama-French, AQR 学术论文中的经典因子

    核心理念:
    - 因子必须有经济学直觉/理论支持
    - 跨资产类别/跨市场/长时间段 robust
    - 重视交易成本和容量
    """

    def __init__(self, data: dict):
        self.data = data  # {'price': df, 'fundamental': df, ...}

    # ========== VALUE FACTORS ==========

    def book_to_market(self) -> pd.DataFrame:
        """
        HML (Fama-French)
        AQR改进: 使用最新的价格但PIT的book value
        """
        bv = self.data['book_value']  # PIT
        mcap = self.data['market_cap']
        bm = bv / mcap
        return self._winsorize_and_standardize(bm)

    def earnings_yield(self) -> pd.DataFrame:
        """
        E/P: 盈利收益率
        AQR偏好用多种盈利指标的复合
        """
        # 使用trailing 12m earnings
        earnings_ttm = self.data['earnings_ttm']  # PIT
        price = self.data['close']
        ep = earnings_ttm / (price * self.data['shares_outstanding'])
        return self._winsorize_and_standardize(ep)

    def composite_value(self) -> pd.DataFrame:
        """
        AQR综合价值因子: 多个价值指标等权复合
        - Book/Price
        - Earnings/Price
        - Cashflow/Price
        - Sales/Price (Lakonishok et al.)
        - 5yr avg Earnings/Price
        """
        factors = [
            self.book_to_market(),
            self.earnings_yield(),
            self._cashflow_yield(),
            self._sales_yield(),
            self._avg_earnings_yield(years=5),
        ]
        composite = sum(f.rank(axis=1, pct=True) for f in factors) / len(factors)
        return self._winsorize_and_standardize(composite)

    # ========== MOMENTUM FACTORS ==========

    def price_momentum_12_1(self) -> pd.DataFrame:
        """
        UMD (Jegadeesh & Titman, 1993)
        过去12个月收益率，跳过最近1个月
        """
        price = self.data['close']
        mom = price.shift(21) / price.shift(252) - 1  # t-1月 to t-12月
        return self._winsorize_and_standardize(mom)

    def time_series_momentum(self) -> pd.DataFrame:
        """
        AQR (Moskowitz, Ooi, Pedersen, 2012)
        时间序列动量: 看自身过去表现，而非截面排名
        """
        returns = self.data['returns']
        signal = returns.rolling(252).mean() / returns.rolling(252).std()
        return signal

    def industry_momentum(self) -> pd.DataFrame:
        """
        行业动量 (Moskowitz & Grinblatt, 1999)
        先算行业平均动量，再分配给行业内个股
        """
        price = self.data['close']
        industries = self.data['industry']
        ret_12_1 = price.shift(21) / price.shift(252) - 1

        ind_mom = pd.DataFrame(index=ret_12_1.index, columns=ret_12_1.columns)
        for date in ret_12_1.index:
            for ind in industries.loc[date].unique():
                mask = industries.loc[date] == ind
                ind_avg = ret_12_1.loc[date, mask].mean()
                ind_mom.loc[date, mask] = ind_avg

        return self._winsorize_and_standardize(ind_mom)

    def residual_momentum(self) -> pd.DataFrame:
        """
        残差动量 (Blitz, Huij, Martens, 2011)
        剔除Fama-French因子暴露后的个股alpha的动量
        比传统动量更 robust，受crash risk影响更小
        """
        returns = self.data['returns']
        mkt = self.data['market_return']
        smb = self.data['smb_factor']
        hml = self.data['hml_factor']

        residuals = pd.DataFrame(index=returns.index, columns=returns.columns)
        window = 252

        for col in returns.columns:
            for i in range(window, len(returns)):
                y = returns[col].iloc[i - window:i].values
                X = np.column_stack([
                    np.ones(window),
                    mkt.iloc[i - window:i].values,
                    smb.iloc[i - window:i].values,
                    hml.iloc[i - window:i].values,
                ])
                try:
                    beta = np.linalg.lstsq(X, y, rcond=None)[0]
                    resid = y - X @ beta
                    # 残差动量: 残差的t统计量
                    residuals.iloc[i][col] = resid.mean() / (resid.std() / np.sqrt(window))
                except Exception:
                    residuals.iloc[i][col] = np.nan

        return self._winsorize_and_standardize(residuals)

    # ========== QUALITY FACTORS ==========

    def quality_minus_junk(self) -> pd.DataFrame:
        """
        QMJ (Asness, Frazzini, Pedersen, 2019)
        质量 = 盈利能力 + 成长性 + 安全性 + 支付能力

        这是 AQR 最著名的因子之一
        """
        profitability = self._profitability_score()
        growth = self._growth_score()
        safety = self._safety_score()
        payout = self._payout_score()

        qmj = (profitability + growth + safety + payout) / 4.0
        return self._winsorize_and_standardize(qmj)

    def _profitability_score(self) -> pd.DataFrame:
        """
        盈利能力子因子:
        - GPOA (Gross Profit / Assets)
        - ROE
        - ROA
        - CFOA (Cash Flow / Assets)
        - Gross Margin
        - Accruals (low is better)
        """
        gpoa = self.data['gross_profit'] / self.data['total_assets']
        roe = self.data['net_income'] / self.data['book_equity']
        roa = self.data['net_income'] / self.data['total_assets']
        cfoa = self.data['operating_cf'] / self.data['total_assets']
        gmar = self.data['gross_profit'] / self.data['revenue']

        accruals = (self.data['net_income'] - self.data['operating_cf']) / self.data['total_assets']
        acc_rank = (-accruals).rank(axis=1, pct=True)  # 低应计利润更好

        components = [gpoa, roe, roa, cfoa, gmar, acc_rank]
        return sum(c.rank(axis=1, pct=True) for c in components) / len(components)

    def _growth_score(self) -> pd.DataFrame:
        """
        成长性子因子:
        - 5年盈利增长率
        - 5年收入增长率
        - GPOA变化
        """
        # 简化实现
        earn_growth = self.data['earnings_ttm'] / self.data['earnings_ttm'].shift(252 * 5) - 1
        rev_growth = self.data['revenue_ttm'] / self.data['revenue_ttm'].shift(252 * 5) - 1
        return (earn_growth.rank(axis=1, pct=True) +
                rev_growth.rank(axis=1, pct=True)) / 2

    def _safety_score(self) -> pd.DataFrame:
        """
        安全性子因子:
        - 低Beta
        - 低杠杆 (D/E)
        - 低盈利波动
        - 低破产概率 (Altman Z-score / Ohlson O-score)
        """
        beta = self.data['beta_252d']
        leverage = self.data['total_debt'] / self.data['book_equity']
        earn_vol = self.data['earnings_ttm'].rolling(252 * 5).std()

        return ((-beta).rank(axis=1, pct=True) +
                (-leverage).rank(axis=1, pct=True) +
                (-earn_vol).rank(axis=1, pct=True)) / 3

    def _payout_score(self) -> pd.DataFrame:
        """
        支付能力:
        - 股息支付率
        - 净回购率
        - 净发行 (低发行更好)
        """
        div_yield = self.data['dividend'] / self.data['close']
        buyback_yield = self.data['net_buyback'] / self.data['market_cap']
        net_issuance = -self.data['share_issuance'] / self.data['market_cap']

        return (div_yield.rank(axis=1, pct=True) +
                buyback_yield.rank(axis=1, pct=True) +
                net_issuance.rank(axis=1, pct=True)) / 3

    # ========== LOW-RISK / BAB FACTORS ==========

    def betting_against_beta(self) -> pd.DataFrame:
        """
        BAB (Frazzini & Pedersen, 2014)
        做多低Beta股票（带杠杆），做空高Beta股票（去杠杆）
        使核心的杠杆约束理论
        """
        beta = self.data['beta_252d']

        # 排名标准化为信号
        beta_rank = beta.rank(axis=1, pct=True)

        # BAB信号: 低beta为正，高beta为负
        signal = -(beta_rank - 0.5)

        # 杠杆调整: 多头杠杆 = 1/avg(beta_low), 空头杠杆 = 1/avg(beta_high)
        return self._winsorize_and_standardize(signal)

    # ========== UTILITY FUNCTIONS ==========

    @staticmethod
    def _winsorize_and_standardize(df: pd.DataFrame,
                                    clip_std: float = 3.0) -> pd.DataFrame:
        """标准化处理"""
        # 截面去极值
        mean = df.mean(axis=1)
        std = df.std(axis=1)
        upper = mean + clip_std * std
        lower = mean - clip_std * std

        clipped = df.clip(lower=lower, upper=upper, axis=0)

        # 截面标准化
        result = clipped.sub(clipped.mean(axis=1), axis=0)
        result = result.div(result.std(axis=1), axis=0)

        return result
```

---

## 四、Alpha 评估与筛选

### 4.1 综合评估框架

```python
from scipy import stats
from typing import Optional


class AlphaEvaluator:
    """
    Alpha 综合评估框架

    顶级基金评估alpha的核心维度:
    1. 预测能力 (IC/IR)
    2. PnL表现 (Sharpe/Drawdown)
    3. 换手率与交易成本
    4. 容量 (Capacity)
    5. 稳定性 (Decay/Regime)
    6. 独立性 (与已有alpha的相关性)
    """

    def __init__(self, alpha_values: pd.DataFrame,
                 forward_returns: pd.DataFrame,
                 market_data: dict):
        self.alpha = alpha_values
        self.forward_returns = forward_returns
        self.market_data = market_data

    def full_evaluation(self) -> dict:
        """完整评估报告"""
        report = {}
        report['predictive_power'] = self._evaluate_predictive_power()
        report['pnl_metrics'] = self._evaluate_pnl()
        report['turnover_analysis'] = self._evaluate_turnover()
        report['capacity'] = self._evaluate_capacity()
        report['stability'] = self._evaluate_stability()
        report['risk_analysis'] = self._evaluate_risk()
        report['overall_score'] = self._compute_overall_score(report)
        return report

    def _evaluate_predictive_power(self) -> dict:
        """
        预测能力评估
        IC (Information Coefficient) 是最核心的指标
        """
        # 每日 Rank IC
        daily_ic = []
        daily_ic_pearson = []

        for date in self.alpha.index:
            if date not in self.forward_returns.index:
                continue
            a = self.alpha.loc[date]
            r = self.forward_returns.loc[date]
            mask = a.notna() & r.notna()
            if mask.sum() < 30:
                continue

            # Spearman Rank IC (更稳健)
            rank_ic = a[mask].corr(r[mask], method='spearman')
            daily_ic.append(rank_ic)

            # Pearson IC
            pearson_ic = a[mask].corr(r[mask], method='pearson')
            daily_ic_pearson.append(pearson_ic)

        ic_series = pd.Series(daily_ic)
        ic_pearson = pd.Series(daily_ic_pearson)

        # IC分析
        avg_ic = ic_series.mean()
        ic_std = ic_series.std()
        ir = avg_ic / ic_std if ic_std > 0 else 0
        ic_positive_ratio = (ic_series > 0).mean()

        # IC自相关 (IC decay)
        ic_autocorr = {}
        for lag in [1, 5, 10, 20]:
            ic_autocorr[f'lag_{lag}'] = ic_series.autocorr(lag=lag)

        # 分位数分析
        quantile_returns = self._quantile_analysis()

        # IC的统计显著性
        t_stat = avg_ic / (ic_std / np.sqrt(len(ic_series)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(ic_series) - 1))

        return {
            'rank_ic_mean': avg_ic,
            'rank_ic_std': ic_std,
            'information_ratio': ir,
            'ic_positive_ratio': ic_positive_ratio,
            'pearson_ic_mean': ic_pearson.mean(),
            'ic_t_stat': t_stat,
            'ic_p_value': p_value,
            'ic_autocorrelation': ic_autocorr,
            'quantile_returns': quantile_returns,
            'ic_series': ic_series,
        }

    def _quantile_analysis(self, n_quantiles=10) -> pd.DataFrame:
        """
        分位数收益分析 (Decile Analysis)
        检验因子的单调性
        """
        results = []

        for date in self.alpha.index:
            if date not in self.forward_returns.index:
                continue
            a = self.alpha.loc[date]
            r = self.forward_returns.loc[date]
            mask = a.notna() & r.notna()

            if mask.sum() < n_quantiles * 10:
                continue

            # 分位数分组
            quantile_labels = pd.qcut(a[mask], n_quantiles,
                                       labels=range(1, n_quantiles + 1),
                                       duplicates='drop')

            for q in range(1, n_quantiles + 1):
                q_mask = quantile_labels == q
                if q_mask.sum() > 0:
                    results.append({
                        'date': date,
                        'quantile': q,
                        'mean_return': r[mask][q_mask].mean(),
                        'count': q_mask.sum(),
                    })

        results_df = pd.DataFrame(results)
        if results_df.empty:
            return pd.DataFrame()

        # 汇总
        summary = results_df.groupby('quantile').agg({
            'mean_return': ['mean', 'std'],
            'count': 'mean',
        })

        # 计算 long-short spread
        q_high = results_df[results_df['quantile'] == n_quantiles]['mean_return'].mean()
        q_low = results_df[results_df['quantile'] == 1]['mean_return'].mean()
        summary.attrs['long_short_spread'] = q_high - q_low

        # 单调性检验 (Spearman correlation of quantile vs return)
        avg_by_q = results_df.groupby('quantile')['mean_return'].mean()
        monotonicity = stats.spearmanr(range(len(avg_by_q)), avg_by_q.values)[0]
        summary.attrs['monotonicity'] = monotonicity

        return summary

    def _evaluate_pnl(self) -> dict:
        """
        PnL回测评估
        构建多空等权组合
        """
        # 生成每日持仓权重
        weights = self.alpha.rank(axis=1, pct=True) - 0.5
        weights = weights.div(weights.abs().sum(axis=1), axis=0)  # 归一化

        # 计算组合收益
        portfolio_returns = (weights.shift(1) * self.forward_returns).sum(axis=1)
        portfolio_returns = portfolio_returns.dropna()

        # 基本指标
        ann_return = portfolio_returns.mean() * 252
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # 月度胜率
        monthly_returns = portfolio_returns.resample('M').sum()
        win_rate = (monthly_returns > 0).mean()

        # 偏度和峰度
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()

        # 多头/空头分解
        long_weights = weights.clip(lower=0)
        short_weights = weights.clip(upper=0)
        long_returns = (long_weights.shift(1) * self.forward_returns).sum(axis=1)
        short_returns = (short_weights.shift(1) * self.forward_returns).sum(axis=1)

        return {
            'annual_return': ann_return,
            'annual_volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'monthly_win_rate': win_rate,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'long_return': long_returns.mean() * 252,
            'short_return': short_returns.mean() * 252,
            'pnl_series': portfolio_returns,
        }

    def _evaluate_turnover(self) -> dict:
        """
        换手率评估
        决定了alpha扣除交易成本后是否还有效
        """
        weights = self.alpha.rank(axis=1, pct=True) - 0.5
        weights = weights.div(weights.abs().sum(axis=1), axis=0)

        # 日换手率
        daily_turnover = weights.diff().abs().sum(axis=1) / 2
        avg_daily_turnover = daily_turnover.mean()
        annual_turnover = avg_daily_turnover * 252

        # 估算交易成本
        # 大型基金的交易成本通常为 5-20bps 单边
        cost_scenarios = {}
        for cost_bps in [5, 10, 15, 20, 30]:
            cost_per_trade = cost_bps / 10000
            daily_cost = avg_daily_turnover * cost_per_trade * 2  # 双边
            annual_cost = daily_cost * 252

            pnl = self._evaluate_pnl()
            net_return = pnl['annual_return'] - annual_cost
            net_sharpe = net_return / pnl['annual_volatility'] if pnl['annual_volatility'] > 0 else 0

            cost_scenarios[f'{cost_bps}bps'] = {
                'annual_cost': annual_cost,
                'net_return': net_return,
                'net_sharpe': net_sharpe,
            }

        # 换手率归因
        holding_period = 1 / (avg_daily_turnover + 1e-10)  # 估算平均持仓天数

        return {
            'avg_daily_turnover': avg_daily_turnover,
            'annual_turnover': annual_turnover,
            'implied_holding_period_days': holding_period,
            'cost_scenarios': cost_scenarios,
            'breakeven_cost_bps': self._breakeven_cost(weights),
        }

    def _breakeven_cost(self, weights):
        """计算盈亏平衡交易成本"""
        turnover = weights.diff().abs().sum(axis=1).mean()
        pnl = self._evaluate_pnl()
        if turnover > 0:
            return (pnl['annual_return'] / (turnover * 252 * 2)) * 10000
        return np.inf

    def _evaluate_capacity(self) -> dict:
        """
        容量评估
        大型基金（AUM > $10B）非常关注
        """
        weights = self.alpha.rank(axis=1, pct=True) - 0.5
        weights = weights.div(weights.abs().sum(axis=1), axis=0)

        adv = self.market_data.get('adv20')
        if adv is None:
            return {'estimated_capacity': 'N/A'}

        # 参与率约束: 每只股票交易量不超过ADV的某个比例
        participation_rates = [0.01, 0.02, 0.05, 0.10]
        capacity_estimates = {}

        for pr in participation_rates:
            # 每只股票最大可交易金额 = ADV * 参与率
            max_trade = adv * pr
            # 将权重转换为金额需要的AUM
            abs_weights = weights.abs()
            # 对于给定AUM, 每只股票的交易量 = AUM * |weight_change|
            # 约束: AUM * |dw| <= ADV * pr
            weight_changes = weights.diff().abs()
            # 最大AUM = min(ADV * pr / |dw|) over all stocks
            constraint = max_trade / (weight_changes + 1e-10)
            daily_capacity = constraint.min(axis=1)
            capacity_estimates[f'pr_{int(pr * 100)}pct'] = daily_capacity.median()

        return {
            'capacity_by_participation': capacity_estimates,
            'avg_stock_adv': adv.mean().mean() if adv is not None else None,
        }

    def _evaluate_stability(self) -> dict:
        """
        稳定性评估
        检查alpha是否在不同时间段/市场环境下稳定
        """
        pnl = self._evaluate_pnl()
        portfolio_returns = pnl['pnl_series']

        # 滚动Sharpe
        rolling_sharpe = (
            portfolio_returns.rolling(252).mean() /
            portfolio_returns.rolling(252).std() * np.sqrt(252)
        )

        # 按年份分析
        annual_metrics = {}
        for year in portfolio_returns.index.year.unique():
            year_returns = portfolio_returns[portfolio_returns.index.year == year]
            if len(year_returns) > 50:
                ann_ret = year_returns.mean() * 252
                ann_vol = year_returns.std() * np.sqrt(252)
                annual_metrics[year] = {
                    'return': ann_ret,
                    'sharpe': ann_ret / ann_vol if ann_vol > 0 else 0,
                }

        # IC decay分析
        ic_data = self._evaluate_predictive_power()
        ic_series = ic_data['ic_series']

        # 前半段 vs 后半段
        mid = len(ic_series) // 2
        first_half_ir = ic_series.iloc[:mid].mean() / ic_series.iloc[:mid].std()
        second_half_ir = ic_series.iloc[mid:].mean() / ic_series.iloc[mid:].std()

        # 市场环境分析 (牛市/熊市)
        market_return = self.market_data.get('market_return')
        if market_return is not None:
            bull = market_return > 0
            bear = market_return <= 0
            bull_returns = portfolio_returns[bull.reindex(portfolio_returns.index, fill_value=False)]
            bear_returns = portfolio_returns[bear.reindex(portfolio_returns.index, fill_value=False)]

            regime_analysis = {
                'bull_sharpe': (bull_returns.mean() * 252) / (bull_returns.std() * np.sqrt(252))
                              if len(bull_returns) > 20 else np.nan,
                'bear_sharpe': (bear_returns.mean() * 252) / (bear_returns.std() * np.sqrt(252))
                              if len(bear_returns) > 20 else np.nan,
            }
        else:
            regime_analysis = {}

        return {
            'rolling_sharpe_std': rolling_sharpe.std(),
            'annual_metrics': annual_metrics,
            'first_half_ir': first_half_ir,
            'second_half_ir': second_half_ir,
            'ir_decay_ratio': second_half_ir / first_half_ir if first_half_ir != 0 else np.nan,
            'regime_analysis': regime_analysis,
        }

    def _evaluate_risk(self) -> dict:
        """风险分析"""
        pnl = self._evaluate_pnl()
        returns = pnl['pnl_series']

        # VaR 和 CVaR
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()

        # 尾部风险
        left_tail = returns.quantile(0.01)
        right_tail = returns.quantile(0.99)
        tail_ratio = abs(right_tail / left_tail) if left_tail != 0 else np.nan

        # 与常见因子的相关性
        factor_correlations = {}
        for factor_name in ['market', 'smb', 'hml', 'momentum', 'quality']:
            factor_data = self.market_data.get(f'{factor_name}_return')
            if factor_data is not None:
                aligned = pd.concat([returns, factor_data], axis=1).dropna()
                if len(aligned) > 50:
                    factor_correlations[factor_name] = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

        return {
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_ratio': tail_ratio,
            'factor_correlations': factor_correlations,
        }

    def _compute_overall_score(self, report: dict) -> dict:
        """
        综合评分 — 模拟顶级基金的alpha准入门槛
        """
        pp = report['predictive_power']
        pnl = report['pnl_metrics']
        to = report['turnover_analysis']
        stab = report['stability']

        # 各维度评分 (0-10)
        ic_score = min(10, max(0, pp['information_ratio'] * 5))
        sharpe_score = min(10, max(0, pnl['sharpe_ratio'] * 3))
        drawdown_score = min(10, max(0, (1 + pnl['max_drawdown']) * 10))
        stability_score = min(10, max(0, 10 - stab['rolling_sharpe_std'] * 5))

        # 交易成本后的Sharpe
        net_sharpe = to['cost_scenarios'].get('10bps', {}).get('net_sharpe', 0)
        net_score = min(10, max(0, net_sharpe * 3))

        # 是否通过
        THRESHOLDS = {
            'min_ir': 0.05,          # IR > 0.05
            'min_ic': 0.02,          # |IC| > 2%
            'min_sharpe': 0.5,       # Sharpe > 0.5
            'max_drawdown': -0.20,   # MaxDD > -20%
            'min_net_sharpe': 0.3,   # Net Sharpe > 0.3 (after 10bps cost)
            'ic_t_stat': 2.0,        # t-stat > 2.0
        }

        passed = all([
            pp['information_ratio'] > THRESHOLDS['min_ir'],
            abs(pp['rank_ic_mean']) > THRESHOLDS['min_ic'],
            pnl['sharpe_ratio'] > THRESHOLDS['min_sharpe'],
            pnl['max_drawdown'] > THRESHOLDS['max_drawdown'],
            net_sharpe > THRESHOLDS['min_net_sharpe'],
            abs(pp['ic_t_stat']) > THRESHOLDS['ic_t_stat'],
        ])

        return {
            'ic_score': ic_score,
            'sharpe_score': sharpe_score,
            'drawdown_score': drawdown_score,
            'stability_score': stability_score,
            'net_performance_score': net_score,
            'overall_score': (ic_score + sharpe_score + drawdown_score +
                            stability_score + net_score) / 5,
            'passed_threshold': passed,
            'thresholds': THRESHOLDS,
        }
```

---

## 五、Alpha 组合与正交化

### 5.1 Alpha 正交化引擎

```python
from sklearn.linear_model import LinearRegression, Lasso


class AlphaOrthogonalizer:
    """
    Alpha正交化引擎

    核心问题: 新发现的alpha可能与现有alpha高度相关
    需要提取"增量信息"

    方法:
    1. Gram-Schmidt正交化
    2. 回归残差法
    3. PCA降维后投影
    """

    def __init__(self, existing_alphas: Dict[str, pd.DataFrame]):
        """
        existing_alphas: 已有的alpha信号字典
        """
        self.existing_alphas = existing_alphas

    def compute_alpha_correlation_matrix(self) -> pd.DataFrame:
        """
        计算alpha间的相关性矩阵
        使用IC级别的相关性，而非原始信号相关性
        """
        alpha_names = list(self.existing_alphas.keys())
        n = len(alpha_names)
        corr_matrix = pd.DataFrame(
            np.zeros((n, n)),
            index=alpha_names,
            columns=alpha_names
        )

        for i in range(n):
            for j in range(i, n):
                # 截面上计算相关性，然后取时间平均
                alpha_i = self.existing_alphas[alpha_names[i]]
                alpha_j = self.existing_alphas[alpha_names[j]]

                daily_corrs = []
                for date in alpha_i.index:
                    if date in alpha_j.index:
                        ai = alpha_i.loc[date].dropna()
                        aj = alpha_j.loc[date].dropna()
                        common = ai.index.intersection(aj.index)
                        if len(common) > 30:
                            corr = ai[common].corr(aj[common], method='spearman')
                            daily_corrs.append(corr)

                avg_corr = np.mean(daily_corrs) if daily_corrs else 0
                corr_matrix.iloc[i, j] = avg_corr
                corr_matrix.iloc[j, i] = avg_corr

        return corr_matrix

    def orthogonalize_new_alpha(self, new_alpha: pd.DataFrame,
                                 method: str = 'regression') -> pd.DataFrame:
        """
        将新alpha相对于已有alpha正交化

        method:
        - 'regression': OLS回归取残差
        - 'gram_schmidt': Gram-Schmidt正交化
        - 'lasso': L1正则化回归取残差（更稳健）
        """
        if method == 'regression':
            return self._orthogonalize_regression(new_alpha)
        elif method == 'gram_schmidt':
            return self._orthogonalize_gram_schmidt(new_alpha)
        elif method == 'lasso':
            return self._orthogonalize_lasso(new_alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _orthogonalize_regression(self, new_alpha: pd.DataFrame) -> pd.DataFrame:
        """
        截面回归正交化
        对每个交易日:
        new_alpha_i = β₁·alpha1_i + β₂·alpha2_i + ... + ε_i
        返回 ε (残差) 作为正交化后的alpha
        """
        residuals = pd.DataFrame(
            index=new_alpha.index,
            columns=new_alpha.columns,
            dtype=float
        )

        for date in new_alpha.index:
            y = new_alpha.loc[date].values
            X_list = []
            for alpha_name, alpha_df in self.existing_alphas.items():
                if date in alpha_df.index:
                    X_list.append(alpha_df.loc[date].values)

            if not X_list:
                residuals.loc[date] = y
                continue

            X = np.column_stack(X_list)

            # 去除NaN
            mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))
            if mask.sum() < X.shape[1] + 10:
                residuals.loc[date] = np.nan
                continue

            reg = LinearRegression(fit_intercept=True)
            reg.fit(X[mask], y[mask])
            y_pred = reg.predict(X[mask])
            res = np.full_like(y, np.nan)
            res[mask] = y[mask] - y_pred

            residuals.loc[date] = res

        return residuals

    def _orthogonalize_lasso(self, new_alpha: pd.DataFrame,
                              alpha_reg: float = 0.01) -> pd.DataFrame:
        """
        LASSO正交化 — 当已有alpha数量多时更稳健
        自动选择哪些alpha是真正相关的
        """
        residuals = pd.DataFrame(
            index=new_alpha.index,
            columns=new_alpha.columns,
            dtype=float
        )

        for date in new_alpha.index:
            y = new_alpha.loc[date].values
            X_list = []
            for alpha_name, alpha_df in self.existing_alphas.items():
                if date in alpha_df.index:
                    X_list.append(alpha_df.loc[date].values)

            if not X_list:
                residuals.loc[date] = y
                continue

            X = np.column_stack(X_list)
            mask = ~(np.isnan(y) | np.any(np.isnan(X), axis=1))

            if mask.sum() < X.shape[1] + 10:
                residuals.loc[date] = np.nan
                continue

            # 标准化
            X_std = (X[mask] - X[mask].mean(axis=0)) / (X[mask].std(axis=0) + 1e-10)
            y_std = (y[mask] - y[mask].mean()) / (y[mask].std() + 1e-10)

            reg = Lasso(alpha=alpha_reg, fit_intercept=False, max_iter=1000)
            reg.fit(X_std, y_std)
            y_pred = reg.predict(X_std)
            res = np.full_like(y, np.nan)
            res[mask] = y[mask] - (y_pred * y[mask].std() + y[mask].mean())

            residuals.loc[date] = res

        return residuals

    def incremental_ic(self, new_alpha: pd.DataFrame,
                       forward_returns: pd.DataFrame) -> dict:
        """
        增量IC分析
        衡量新alpha在控制已有alpha后的边际预测能力
        """
        # 原始IC
        raw_ic = self._compute_avg_ic(new_alpha, forward_returns)

        # 正交化后IC
        orthogonal_alpha = self.orthogonalize_new_alpha(new_alpha, method='regression')
        orthogonal_ic = self._compute_avg_ic(orthogonal_alpha, forward_returns)

        # IC比例
        ic_retention = abs(orthogonal_ic) / abs(raw_ic) if raw_ic != 0 else 0

        return {
            'raw_ic': raw_ic,
            'orthogonal_ic': orthogonal_ic,
            'ic_retention_ratio': ic_retention,
            'is_incremental': abs(orthogonal_ic) > 0.01,  # 阈值
        }

    def _compute_avg_ic(self, alpha: pd.DataFrame,
                        forward_returns: pd.DataFrame) -> float:
        """计算平均IC"""
        ics = []
        for date in alpha.index:
            if date in forward_returns.index:
                a = alpha.loc[date]
                r = forward_returns.loc[date]
                mask = a.notna() & r.notna()
                if mask.sum() > 30:
                    ic = a[mask].corr(r[mask], method='spearman')
                    ics.append(ic)
        return np.mean(ics) if ics else 0


class AlphaCombiner:
    """
    Alpha组合优化器

    将多个alpha信号组合为最优的综合信号

    方法:
    1. 等权 (Naive)
    2. IC加权
    3. IC_IR加权
    4. 最优化 (Mean-Variance on IC)
    5. 机器学习 (Ridge/Elastic Net)
    """

    def __init__(self, alphas: Dict[str, pd.DataFrame],
                 forward_returns: pd.DataFrame):
        self.alphas = alphas
        self.forward_returns = forward_returns
        self.alpha_names = list(alphas.keys())

    def equal_weight(self) -> pd.DataFrame:
        """等权组合"""
        combined = None
        for name, alpha in self.alphas.items():
            ranked = alpha.rank(axis=1, pct=True) - 0.5
            if combined is None:
                combined = ranked
            else:
                combined = combined + ranked
        return combined / len(self.alphas)

    def ic_weighted(self, lookback: int = 60) -> pd.DataFrame:
        """
        IC加权: 用历史IC作为权重
        更近期IC高的alpha获得更大权重
        """
        # 计算每个alpha的滚动IC
        rolling_ics = {}
        for name, alpha in self.alphas.items():
            daily_ic = pd.Series(index=alpha.index, dtype=float)
            for date in alpha.index:
                if date in self.forward_returns.index:
                    a = alpha.loc[date]
                    r = self.forward_returns.loc[date]
                    mask = a.notna() & r.notna()
                    if mask.sum() > 30:
                        daily_ic[date] = a[mask].corr(r[mask], method='spearman')
            rolling_ics[name] = daily_ic.rolling(lookback).mean()

        # IC加权组合
        combined = pd.DataFrame(0, index=list(self.alphas.values())[0].index,
                               columns=list(self.alphas.values())[0].columns)

        for date in combined.index:
            weights = {}
            total_weight = 0
            for name in self.alpha_names:
                ic = rolling_ics[name].get(date, 0)
                if np.isnan(ic):
                    ic = 0
                weights[name] = max(ic, 0)  # 只用正IC
                total_weight += weights[name]

            if total_weight > 0:
                for name in self.alpha_names:
                    w = weights[name] / total_weight
                    ranked = self.alphas[name].loc[date].rank(pct=True) - 0.5
                    combined.loc[date] += w * ranked

        return combined

    def optimized_combination(self, lookback: int = 252,
                               shrinkage: float = 0.5) -> pd.DataFrame:
        """
        最优化组合 (Markowitz on alpha signals)

        max  w' * μ_IC
        s.t. w' * Σ_IC * w <= target_risk
             w >= 0
             sum(w) = 1

        使用收缩估计量提升稳定性
        """
        from scipy.optimize import minimize

        # 计算IC时间序列
        ic_matrix = pd.DataFrame()
        for name, alpha in self.alphas.items():
            daily_ic = []
            dates = []
            for date in alpha.index:
                if date in self.forward_returns.index:
                    a = alpha.loc[date]
                    r = self.forward_returns.loc[date]
                    mask = a.notna() & r.notna()
                    if mask.sum() > 30:
                        ic = a[mask].corr(r[mask], method='spearman')
                        daily_ic.append(ic)
                        dates.append(date)
            ic_matrix[name] = pd.Series(daily_ic, index=dates)

        ic_matrix = ic_matrix.dropna()

        # 收缩估计
        mu = ic_matrix.mean().values
        Sigma = ic_matrix.cov().values
        n = len(mu)

        # Ledoit-Wolf 收缩
        Sigma_shrunk = shrinkage * np.diag(np.diag(Sigma)) + (1 - shrinkage) * Sigma

        # 优化
        def neg_ir(w):
            port_ic = w @ mu
            port_var = w @ Sigma_shrunk @ w
            return -(port_ic / np.sqrt(port_var + 1e-10))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n
        w0 = np.ones(n) / n

        result = minimize(neg_ir, w0, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        optimal_weights = result.x

        # 生成组合信号
        combined = pd.DataFrame(0, index=list(self.alphas.values())[0].index,
                               columns=list(self.alphas.values())[0].columns)
        for i, name in enumerate(self.alpha_names):
            ranked = self.alphas[name].rank(axis=1, pct=True) - 0.5
            combined += optimal_weights[i] * ranked

        return combined
```

---

## 六、组合构建与风险管理

### 6.1 组合优化器

```python
from scipy.optimize import minimize


class PortfolioOptimizer:
    """
    组合优化器

    从alpha信号到实际持仓权重
    AQR / Two Sigma 的核心竞争力之一

    关键约束:
    - 风险预算 (风险模型约束)
    - 行业/国家中性
    - 换手率约束 (减少交易成本)
    - 流动性约束
    - 做空限制
    - 个股/行业集中度限制
    """

    def __init__(self, risk_model, transaction_cost_model):
        self.risk_model = risk_model
        self.tc_model = transaction_cost_model

    def optimize(self,
                 alpha_signal: pd.Series,
                 current_portfolio: pd.Series,
                 universe: list,
                 constraints: dict) -> pd.Series:
        """
        单期组合优化

        目标: max  α'w - λ₁·w'Σw - λ₂·TC(w, w_prev)
        s.t.  各类约束

        Parameters:
        -----------
        alpha_signal: 每只股票的alpha值
        current_portfolio: 当前持仓
        universe: 投资宇宙
        constraints: 约束条件字典
        """
        n = len(universe)
        alpha = alpha_signal.reindex(universe).fillna(0).values
        w_prev = current_portfolio.reindex(universe).fillna(0).values

        # 风险模型
        factor_cov = self.risk_model.get_factor_covariance()
        factor_loadings = self.risk_model.get_factor_loadings(universe)
        specific_risk = self.risk_model.get_specific_risk(universe)

        # 投资组合方差 = w' * (B*F*B' + D) * w
        B = factor_loadings  # n x k
        F = factor_cov       # k x k
        D = np.diag(specific_risk ** 2)  # n x n

        def portfolio_variance(w):
            Bw = B.T @ w
            return Bw @ F @ Bw + w @ D @ w

        # 交易成本模型
        def transaction_cost(w):
            return self.tc_model.estimate_cost(w - w_prev, universe)

        # 目标函数
        risk_aversion = constraints.get('risk_aversion', 1.0)
        tc_aversion = constraints.get('tc_aversion', 1.0)

        def objective(w):
            expected_alpha = alpha @ w
            risk = portfolio_variance(w)
            tc = transaction_cost(w)
            return -(expected_alpha - risk_aversion * risk - tc_aversion * tc)

        # ===== 约束条件 =====
        cons = []

        # 1. 美元中性 (多空等额)
        if constraints.get('dollar_neutral', True):
            cons.append({'type': 'eq', 'fun': lambda w: np.sum(w)})

        # 2. 总杠杆约束
        max_leverage = constraints.get('max_gross_leverage', 2.0)
        cons.append({
            'type': 'ineq',
            'fun': lambda w: max_leverage - np.sum(np.abs(w))
        })

        # 3. 行业中性
        if constraints.get('industry_neutral', True):
            industries = self.risk_model.get_industry_membership(universe)
            for ind in np.unique(industries):
                mask = (industries == ind).astype(float)
                cons.append({
                    'type': 'eq',
                    'fun': lambda w, m=mask: m @ w  # 每个行业净暴露为0
                })

        # 4. 因子暴露约束
        max_factor_exposure = constraints.get('max_factor_exposure', 0.5)
        for k in range(B.shape[1]):
            factor_loading = B[:, k]
            cons.append({
                'type': 'ineq',
                'fun': lambda w, fl=factor_loading: max_factor_exposure - abs(fl @ w)
            })

        # 5. 最大换手率
        max_turnover = constraints.get('max_daily_turnover', 0.20)
        cons.append({
            'type': 'ineq',
            'fun': lambda w: max_turnover - np.sum(np.abs(w - w_prev)) / 2
        })

        # 6. 个股权重限制
        max_weight = constraints.get('max_individual_weight', 0.02)
        bounds = [(-max_weight, max_weight)] * n

        # 7. 流动性约束 (不超过ADV的x%)
        if 'adv' in constraints:
            adv = constraints['adv']
            max_participation = constraints.get('max_participation_rate', 0.05)
            for i in range(n):
                max_w = adv[i] * max_participation / constraints.get('portfolio_nav', 1e8)
                bounds[i] = (max(-max_weight, -max_w), min(max_weight, max_w))

        # 求解
        w0 = w_prev.copy()

        result = minimize(
            objective, w0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=universe)
        else:
            # 回退到简单的信号排名方法
            print(f"Optimization failed: {result.message}")
            optimal_weights = pd.Series(alpha, index=universe)
            optimal_weights = optimal_weights.rank(pct=True) - 0.5
            optimal_weights /= optimal_weights.abs().sum()

        return optimal_weights


class TransactionCostModel:
    """
    交易成本模型

    Two Sigma / AQR 使用的多层交易成本估算:
    1. 固定成本 (佣金, 费用)
    2. 价差成本 (bid-ask spread)
    3. 市场冲击 (market impact) — 与交易量和波动率相关
    4. 时间延迟成本 (delay cost)
    """

    def __init__(self, market_data: dict):
        self.market_data = market_data

    def estimate_cost(self, trades: np.ndarray, universe: list,
                      nav: float = 1e8) -> float:
        """
        估算总交易成本

        经典市场冲击模型 (Almgren-Chriss):
        Impact = σ * sqrt(Q / V) * sign(Q)

        Where:
        σ = daily volatility
        Q = trade quantity
        V = daily volume
        """
        total_cost = 0

        for i, ticker in enumerate(universe):
            trade_dollars = abs(trades[i]) * nav

            if trade_dollars < 100:
                continue

            # 1. 固定佣金
            commission = 0.001 * trade_dollars  # 0.1 bps

            # 2. 价差成本
            spread = self.market_data.get('spread', {}).get(ticker, 0.0005)
            spread_cost = 0.5 * spread * trade_dollars

            # 3. 市场冲击 (Square-root model)
            adv_dollars = self.market_data.get('adv_dollars', {}).get(ticker, 1e7)
            volatility = self.market_data.get('volatility', {}).get(ticker, 0.02)

            participation_rate = trade_dollars / adv_dollars
            # Almgren model: impact = η * σ * sqrt(participation_rate)
            eta = 0.5  # 冲击系数
            impact_bps = eta * volatility * np.sqrt(participation_rate)
            impact_cost = impact_bps * trade_dollars

            total_cost += commission + spread_cost + impact_cost

        return total_cost / nav  # 返回占NAV比例


class RiskModel:
    """
    多因子风险模型 (类似 Barra / Axioma)

    协方差估计:
    Σ = B * F * B' + D

    B: 因子载荷矩阵 (n x k)
    F: 因子协方差矩阵 (k x k)
    D: 特异性风险矩阵 (n x n, 对角)
    """

    def __init__(self, n_factors: int = 40):
        self.n_factors = n_factors
        self.factor_returns = None
        self.factor_loadings = None
        self.factor_covariance = None
        self.specific_risk = None

    def estimate(self, returns: pd.DataFrame,
                 factor_exposures: pd.DataFrame):
        """
        估计风险模型

        步骤:
        1. 截面回归获取因子收益
        2. 因子协方差矩阵估计 (带Newey-West调整)
        3. 特异性风险估计 (带结构化调整)
        """
        n_dates = len(returns)
        n_stocks = len(returns.columns)

        # 1. 截面回归: r_i,t = Σ_k β_i,k * f_k,t + ε_i,t
        factor_returns_list = []
        residuals_list = []

        for date in returns.index:
            y = returns.loc[date].dropna()
            X = factor_exposures.loc[date].reindex(y.index).dropna()
            common = y.index.intersection(X.index)

            if len(common) < self.n_factors + 10:
                continue

            y_clean = y[common].values
            X_clean = X.loc[common].values

            # WLS回归 (以市值为权重)
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X_clean, y_clean)

            factor_returns_list.append(
                pd.Series(reg.coef_, index=X.columns, name=date)
            )
            pred = reg.predict(X_clean)
            resid = pd.Series(y_clean - pred, index=common, name=date)
            residuals_list.append(resid)

        self.factor_returns = pd.DataFrame(factor_returns_list)

        # 2. 因子协方差 (Exponential weighted + Newey-West)
        halflife = 90
        weights = np.exp(-np.log(2) * np.arange(len(self.factor_returns))[::-1] / halflife)
        weights /= weights.sum()

        weighted_returns = self.factor_returns.values * np.sqrt(weights[:, np.newaxis])
        self.factor_covariance = weighted_returns.T @ weighted_returns * 252

        # 3. 特异性风险
        residuals_df = pd.DataFrame(residuals_list)
        self.specific_risk = residuals_df.std() * np.sqrt(252)

        # 结构化调整: 将特异性风险与波动率/市值建立联系
        # (Barra USE4 方法)

    def get_factor_covariance(self):
        return self.factor_covariance

    def get_factor_loadings(self, universe):
        return self.factor_loadings

    def get_specific_risk(self, universe):
        return self.specific_risk

    def get_industry_membership(self, universe):
        pass
```

---

## 七、回测引擎

### 7.1 事件驱动回测框架

```python
from collections import defaultdict
from datetime import timedelta


class EventDrivenBacktester:
    """
    事件驱动回测引擎

    顶级基金的回测标准:
    1. Point-in-Time 严格
    2. 无前视偏差
    3. 真实的交易成本
    4. 真实的市场微结构
    5. 考虑借券成本/可借性
    6. 分红/送股/并购/退市处理
    """

    def __init__(self, config: dict):
        self.config = config
        self.portfolio = {}  # ticker -> shares
        self.cash = config.get('initial_cash', 1e8)
        self.nav_history = []
        self.trade_history = []
        self.daily_pnl = []

    def run(self,
            alpha_model,
            risk_model,
            optimizer,
            data_provider,
            start_date: str,
            end_date: str):
        """
        主回测循环
        """
        dates = data_provider.get_trading_dates(start_date, end_date)

        for i, date in enumerate(dates):
            # 1. 获取当前可用数据 (严格PIT)
            current_data = data_provider.get_data_as_of(date)

            # 2. 处理公司事件
            self._process_corporate_actions(date, data_provider)

            # 3. 更新投资宇宙
            universe = data_provider.get_universe(date)

            # 4. 生成alpha信号
            alpha_signal = alpha_model.generate_signal(
                data=current_data,
                universe=universe,
                date=date
            )

            # 5. 计算目标持仓
            current_weights = self._get_current_weights(date, data_provider)
            target_weights = optimizer.optimize(
                alpha_signal=alpha_signal,
                current_portfolio=current_weights,
                universe=universe,
                constraints=self.config.get('constraints', {})
            )

            # 6. 执行交易
            trades = target_weights - current_weights
            executed_trades = self._execute_trades(trades, date, data_provider)

            # 7. 记录
            nav = self._compute_nav(date, data_provider)
            self.nav_history.append({'date': date, 'nav': nav})

            if i > 0:
                prev_nav = self.nav_history[-2]['nav']
                daily_return = (nav - prev_nav) / prev_nav
                self.daily_pnl.append({'date': date, 'return': daily_return})

            # 8. 进度报告
            if i % 63 == 0:  # 每季度报告
                print(f"Date: {date} | NAV: ${nav:,.0f} | "
                      f"Positions: {len([v for v in self.portfolio.values() if v != 0])}")

        return self._generate_report()

    def _execute_trades(self, trades: pd.Series, date, data_provider):
        """
        模拟真实的交易执行

        考虑:
        - 成交量限制 (不超过ADV的x%)
        - 价格影响
        - 下单时间 (VWAP/TWAP)
        - 借券可用性 (做空)
        """
        executed = {}
        for ticker, target_trade_weight in trades.items():
            if abs(target_trade_weight) < 1e-6:
                continue

            price = data_provider.get_price(ticker, date)
            adv_shares = data_provider.get_adv_shares(ticker, date)

            if price is None or np.isnan(price):
                continue

            trade_dollars = target_trade_weight * self._compute_nav(date, data_provider)
            trade_shares = int(trade_dollars / price)

            # 成交量限制
            max_participation = self.config.get('max_participation_rate', 0.05)
            max_shares = int(adv_shares * max_participation)
            trade_shares = np.clip(trade_shares, -max_shares, max_shares)

            # 做空检查
            if trade_shares < 0:
                borrow_available = data_provider.get_borrow_availability(ticker, date)
                if not borrow_available:
                    trade_shares = max(trade_shares, -self.portfolio.get(ticker, 0))

                # 借券成本
                borrow_cost = data_provider.get_borrow_cost(ticker, date)
            else:
                borrow_cost = 0

            # 市场冲击
            volatility = data_provider.get_volatility(ticker, date)
            participation_rate = abs(trade_shares) / max(adv_shares, 1)
            impact = 0.5 * volatility * np.sqrt(participation_rate)

            execution_price = price * (1 + impact * np.sign(trade_shares))

            # 更新持仓
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + trade_shares
            self.cash -= trade_shares * execution_price

            # 佣金
            commission = abs(trade_shares * execution_price) * 0.0001
            self.cash -= commission

            executed[ticker] = {
                'shares': trade_shares,
                'price': execution_price,
                'impact': impact,
                'commission': commission,
            }

            self.trade_history.append({
                'date': date,
                'ticker': ticker,
                **executed[ticker],
            })

        return executed

    def _process_corporate_actions(self, date, data_provider):
        """处理公司行为"""
        actions = data_provider.get_corporate_actions(date)
        for action in actions:
            ticker = action['ticker']
            if ticker not in self.portfolio:
                continue

            if action['type'] == 'split':
                ratio = action['ratio']
                self.portfolio[ticker] = int(self.portfolio[ticker] * ratio)

            elif action['type'] == 'dividend':
                div_per_share = action['amount']
                self.cash += self.portfolio[ticker] * div_per_share

            elif action['type'] == 'delist':
                # 退市: 以最后价格清算
                last_price = action.get('last_price', 0)
                self.cash += self.portfolio[ticker] * last_price
                del self.portfolio[ticker]

            elif action['type'] == 'merger':
                # 并购: 按条款转换
                if action.get('cash_deal'):
                    self.cash += self.portfolio[ticker] * action['cash_per_share']
                    del self.portfolio[ticker]

    def _compute_nav(self, date, data_provider):
        """计算NAV"""
        equity_value = 0
        for ticker, shares in self.portfolio.items():
            price = data_provider.get_price(ticker, date)
            if price is not None and not np.isnan(price):
                equity_value += shares * price
        return self.cash + equity_value

    def _get_current_weights(self, date, data_provider):
        """获取当前持仓权重"""
        nav = self._compute_nav(date, data_provider)
        weights = {}
        for ticker, shares in self.portfolio.items():
            price = data_provider.get_price(ticker, date)
            if price is not None and not np.isnan(price):
                weights[ticker] = shares * price / nav
        return pd.Series(weights)

    def _generate_report(self) -> dict:
        """生成回测报告"""
        nav_df = pd.DataFrame(self.nav_history).set_index('date')
        pnl_df = pd.DataFrame(self.daily_pnl).set_index('date')

        returns = pnl_df['return']

        report = {
            'total_return': (nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0]) - 1,
            'annual_return': returns.mean() * 252,
            'annual_volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self._max_drawdown(nav_df['nav']),
            'total_trades': len(self.trade_history),
            'total_commission': sum(t['commission'] for t in self.trade_history),
            'nav_series': nav_df,
            'return_series': returns,
        }

        return report

    @staticmethod
    def _max_drawdown(nav_series):
        peak = nav_series.cummax()
        drawdown = (nav_series - peak) / peak
        return drawdown.min()
```

---

## 八、机器学习增强

### 8.1 ML Alpha模型

```python
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


class MLAlphaModel:
    """
    机器学习 Alpha 模型

    Two Sigma 大量使用 ML
    关键挑战:
    1. 低信噪比
    2. 非平稳性
    3. 过拟合风险极高
    4. 需要在截面上泛化
    """

    def __init__(self, model_type='gbm'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None

    def prepare_features(self, raw_data: dict,
                         feature_config: list) -> pd.DataFrame:
        """
        特征工程

        黄金法则:
        1. 所有特征必须截面标准化 (rank / z-score)
        2. 去除极值
        3. 处理缺失值
        4. 滞后以避免前视偏差
        """
        features = {}

        for feat_cfg in feature_config:
            name = feat_cfg['name']
            raw = raw_data[feat_cfg['field']]

            # 应用算子
            if 'operator' in feat_cfg:
                transformed = feat_cfg['operator'](raw)
            else:
                transformed = raw

            # 截面排名标准化
            ranked = transformed.rank(axis=1, pct=True)

            # 去极值
            ranked = ranked.clip(0.01, 0.99)

            # 滞后
            lag = feat_cfg.get('lag', 1)
            features[name] = ranked.shift(lag)

        return features

    def train_rolling(self, features: Dict[str, pd.DataFrame],
                      target: pd.DataFrame,
                      train_window: int = 756,  # 3年
                      retrain_frequency: int = 21,  # 每月重训
                      purge_gap: int = 5):  # 训练-测试间隔
        """
        滚动训练 (Rolling Window)

        严格避免前视偏差:
        - 训练集: [t - train_window, t - purge_gap]
        - 验证集: [t - purge_gap, t]
        - 测试集: [t, t + retrain_frequency]

        Purge Gap: 防止target泄露到训练集
        """
        dates = target.index
        predictions = pd.DataFrame(
            index=target.index,
            columns=target.columns,
            dtype=float
        )

        for t in range(train_window + purge_gap, len(dates), retrain_frequency):
            train_end = t - purge_gap
            train_start = max(0, train_end - train_window)

            # 构建训练数据 (stack截面)
            X_train, y_train = self._stack_panel(
                features, target,
                dates[train_start:train_end]
            )

            if len(X_train) < 1000:
                continue

            # 训练模型
            if self.model_type == 'gbm':
                self.model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    max_features=0.7,
                    min_samples_leaf=100,  # 较大的叶子节点 → 防过拟合
                    random_state=42
                )
            elif self.model_type == 'linear':
                from sklearn.linear_model import Ridge
                self.model = Ridge(alpha=10.0)

            self.model.fit(X_train, y_train)

            # 预测 (t to t + retrain_frequency)
            pred_end = min(t + retrain_frequency, len(dates))
            for pred_date in dates[t:pred_end]:
                X_pred = self._get_features_at_date(features, pred_date)
                if X_pred is not None and len(X_pred) > 0:
                    y_pred = self.model.predict(X_pred)
                    predictions.loc[pred_date, X_pred.index] = y_pred

            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance = pd.Series(
                    self.model.feature_importances_,
                    index=list(features.keys())
                ).sort_values(ascending=False)

        return predictions

    def _stack_panel(self, features, target, dates):
        """将面板数据stack为样本 × 特征矩阵"""
        X_list = []
        y_list = []

        for date in dates:
            # 获取当天的截面数据
            feat_row = {}
            valid_tickers = None

            for name, feat_df in features.items():
                if date in feat_df.index:
                    vals = feat_df.loc[date]
                    feat_row[name] = vals
                    if valid_tickers is None:
                        valid_tickers = vals.dropna().index
                    else:
                        valid_tickers = valid_tickers.intersection(vals.dropna().index)

            if valid_tickers is None or len(valid_tickers) < 50:
                continue

            if date in target.index:
                tgt = target.loc[date]
                valid_tickers = valid_tickers.intersection(tgt.dropna().index)

                if len(valid_tickers) < 50:
                    continue

                X_row = pd.DataFrame(
                    {name: feat_row[name][valid_tickers] for name in features.keys()}
                )
                y_row = tgt[valid_tickers]

                X_list.append(X_row)
                y_list.append(y_row)

        if not X_list:
            return pd.DataFrame(), pd.Series()

        X = pd.concat(X_list)
        y = pd.concat(y_list)

        # 去除NaN
        mask = X.notna().all(axis=1) & y.notna()
        return X[mask], y[mask]

    def _get_features_at_date(self, features, date):
        """获取某一天的截面特征"""
        feat_row = {}
        valid_tickers = None

        for name, feat_df in features.items():
            if date in feat_df.index:
                vals = feat_df.loc[date]
                feat_row[name] = vals
                if valid_tickers is None:
                    valid_tickers = vals.dropna().index
                else:
                    valid_tickers = valid_tickers.intersection(vals.dropna().index)

        if valid_tickers is None or len(valid_tickers) < 10:
            return None

        X = pd.DataFrame(
            {name: feat_row[name][valid_tickers] for name in features.keys()},
            index=valid_tickers
        )
        return X.dropna()


class CrossSectionalTransformer(nn.Module):
    """
    截面注意力模型

    Two Sigma 风格: 使用Transformer捕捉股票间的关系

    输入: 每只股票的特征向量
    输出: 预测的alpha值

    关键设计:
    1. 截面注意力 (股票间相互影响)
    2. 时序编码 (捕捉时间模式)
    3. 市场状态条件 (regime-dependent)
    """

    def __init__(self, n_features: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        # 特征投影
        self.feature_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 截面 Transformer 层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 预测头
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # 市场状态条件网络
        self.market_state_encoder = nn.Sequential(
            nn.Linear(10, d_model),  # 10个市场特征
            nn.ReLU(),
        )

    def forward(self, stock_features: torch.Tensor,
                market_features: torch.Tensor = None,
                mask: torch.Tensor = None):
        """
        Args:
            stock_features: (batch, n_stocks, n_features)
            market_features: (batch, 10) 市场级特征
            mask: (batch, n_stocks) padding mask

        Returns:
            alpha_pred: (batch, n_stocks) 预测alpha值
        """
        # 特征投影
        x = self.feature_proj(stock_features)  # (batch, n_stocks, d_model)

        # 加入市场状态条件
        if market_features is not None:
            market_emb = self.market_state_encoder(market_features)  # (batch, d_model)
            market_emb = market_emb.unsqueeze(1).expand_as(x)
            x = x + market_emb

        # 截面Transformer (股票间注意力)
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)

        # 预测
        alpha_pred = self.prediction_head(x).squeeze(-1)  # (batch, n_stocks)

        # 截面标准化 (零均值)
        if mask is not None:
            alpha_pred = alpha_pred - (alpha_pred * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        else:
            alpha_pred = alpha_pred - alpha_pred.mean(dim=1, keepdim=True)

        return alpha_pred


class AlphaTrainer:
    """
    Alpha模型训练器

    关键训练技巧:
    1. IC loss (rank correlation loss)
    2. 分段训练 + Purge
    3. 多任务学习 (预测不同horizon的收益)
    4. 对抗训练 (adversarial for robustness)
    """

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )

    def ic_loss(self, predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        IC Loss: 最大化预测与实际收益的截面rank correlation

        比MSE更适合alpha模型:
        - 我们不关心绝对值，只关心排序
        - 对异常值更鲁棒
        """
        if mask is not None:
            # 只用有效的股票
            batch_losses = []
            for b in range(predictions.shape[0]):
                m = mask[b]
                if m.sum() < 10:
                    continue
                pred = predictions[b][m]
                tgt = targets[b][m]

                # 可微的Spearman近似 (用soft rank)
                pred_rank = self._soft_rank(pred)
                tgt_rank = self._soft_rank(tgt)

                # Pearson correlation of ranks ≈ Spearman
                pred_centered = pred_rank - pred_rank.mean()
                tgt_centered = tgt_rank - tgt_rank.mean()

                ic = (pred_centered * tgt_centered).sum() / (
                    torch.sqrt((pred_centered ** 2).sum() * (tgt_centered ** 2).sum()) + 1e-8
                )
                batch_losses.append(-ic)  # 取负号因为要最大化

            if batch_losses:
                return torch.stack(batch_losses).mean()
            return torch.tensor(0.0)
        else:
            pred_rank = self._soft_rank(predictions)
            tgt_rank = self._soft_rank(targets)
            pred_c = pred_rank - pred_rank.mean(dim=1, keepdim=True)
            tgt_c = tgt_rank - tgt_rank.mean(dim=1, keepdim=True)
            ic = (pred_c * tgt_c).sum(dim=1) / (
                torch.sqrt((pred_c ** 2).sum(dim=1) * (tgt_c ** 2).sum(dim=1)) + 1e-8
            )
            return -ic.mean()

    def _soft_rank(self, x: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """可微的软排名"""
        # 使用sigmoid近似
        n = x.shape[-1]
        x_expanded = x.unsqueeze(-1)
        x_ref = x.unsqueeze(-2)
        # 每个元素大于其他元素的"软概率"
        pairwise = torch.sigmoid((x_expanded - x_ref) / temperature)
        return pairwise.sum(dim=-1) / n

    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            features = batch['features']   # (batch, n_stocks, n_features)
            targets = batch['returns']      # (batch, n_stocks)
            mask = batch.get('mask', None)  # (batch, n_stocks)
            market = batch.get('market_features', None)

            self.optimizer.zero_grad()

            predictions = self.model(features, market, mask)
            loss = self.ic_loss(predictions, targets, mask)

            # 正则化
            l2_reg = sum(p.pow(2).sum() for p in self.model.parameters())
            loss = loss + self.config.get('l2_lambda', 1e-5) * l2_reg

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
```

---

## 九、实战 Pipeline 整合

### 9.1 完整的 Alpha Research Pipeline

```python
class AlphaResearchPipeline:
    """
    完整的Alpha研究流程

    模拟 Two Sigma / AQR / WorldQuant 的实际工作流

    Stage 1: 假设生成 (Idea Generation)
    Stage 2: Alpha构建 (Signal Construction)
    Stage 3: 单因子分析 (Single Alpha Analysis)
    Stage 4: 增量分析 (Incremental Analysis)
    Stage 5: 组合集成 (Portfolio Integration)
    Stage 6: 实盘监控 (Live Monitoring)
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_provider = None  # 数据提供者
        self.alpha_library = {}   # 已有alpha库
        self.risk_model = None    # 风险模型
        self.portfolio_optimizer = None  # 组合优化器

    def stage1_idea_generation(self, hypothesis: str) -> dict:
        """
        Stage 1: 假设生成

        来源:
        - 学术论文 (AQR风格)
        - 算子搜索 (WorldQuant风格)
        - 替代数据探索 (Two Sigma风格)
        - PM直觉 + 定量化
        """
        return {
            'hypothesis': hypothesis,
            'economic_intuition': None,  # 必须有经济学解释
            'expected_horizon': None,     # 预期持仓期
            'expected_capacity': None,    # 预期容量
            'related_literature': [],     # 相关文献
            'data_requirements': [],      # 数据需求
        }

    def stage2_construct_alpha(self, idea: dict) -> pd.DataFrame:
        """
        Stage 2: Alpha信号构建

        输出: 每日 × 每只股票 的alpha矩阵
        """
        # 获取所需数据 (严格PIT)
        data = self.data_provider.get_pit_data(
            fields=idea['data_requirements'],
            start_date=self.config['backtest_start'],
            end_date=self.config['backtest_end'],
        )

        # 构建信号
        raw_signal = self._compute_raw_signal(data, idea)

        # 标准化处理
        processed_signal = self._standardize_signal(raw_signal)

        return processed_signal

    def stage3_single_alpha_analysis(self, alpha: pd.DataFrame) -> dict:
        """
        Stage 3: 单因子分析

        必须通过的检验:
        □ IC > 2% (绝对值)
        □ IR > 0.05
        □ IC t-stat > 2.0
        □ Decile单调性 > 0.8
        □ Sharpe > 0.5 (多空组合)
        □ MaxDD < 20%
        □ 扣除20bps成本后Sharpe > 0.3
        □ 前后半段IR比 > 0.5
        □ 牛熊市都有正收益
        """
        forward_returns = self.data_provider.get_forward_returns(
            horizon=self.config.get('holding_period', 5)
        )

        evaluator = AlphaEvaluator(alpha, forward_returns, self.data_provider.market_data)
        report = evaluator.full_evaluation()

        # 打印评估结果
        self._print_evaluation_report(report)

        return report

    def stage4_incremental_analysis(self, new_alpha: pd.DataFrame) -> dict:
        """
        Stage 4: 增量分析

        关键问题: 这个新alpha是否提供了增量信息？
        """
        orthogonalizer = AlphaOrthogonalizer(self.alpha_library)

        # 与已有alpha的相关性
        corr_matrix = orthogonalizer.compute_alpha_correlation_matrix()

        # 增量IC
        incremental = orthogonalizer.incremental_ic(
            new_alpha,
            self.data_provider.get_forward_returns()
        )

        # 正交化后的alpha
        orthogonal_alpha = orthogonalizer.orthogonalize_new_alpha(
            new_alpha, method='lasso'
        )

        return {
            'correlation_with_existing': corr_matrix,
            'incremental_ic': incremental,
            'orthogonal_alpha': orthogonal_alpha,
            'recommendation': 'ADD' if incremental['is_incremental'] else 'REJECT',
        }

    def stage5_portfolio_integration(self,
                                      approved_alphas: Dict[str, pd.DataFrame]) -> dict:
        """
        Stage 5: 组合集成

        将所有通过的alpha组合为最终的交易信号
        """
        # Alpha组合
        combiner = AlphaCombiner(
            approved_alphas,
            self.data_provider.get_forward_returns()
        )

        # 多种组合方式
        combined_signals = {
            'equal_weight': combiner.equal_weight(),
            'ic_weighted': combiner.ic_weighted(),
            'optimized': combiner.optimized_combination(),
        }

        # 对每种组合方式进行完整回测
        results = {}
        for name, signal in combined_signals.items():
            backtester = EventDrivenBacktester(self.config)
            result = backtester.run(
                alpha_model=SignalBasedAlphaModel(signal),
                risk_model=self.risk_model,
                optimizer=self.portfolio_optimizer,
                data_provider=self.data_provider,
                start_date=self.config['backtest_start'],
                end_date=self.config['backtest_end'],
            )
            results[name] = result

        return results

    def stage6_live_monitoring(self, live_alpha: pd.DataFrame) -> dict:
        """
        Stage 6: 实盘监控

        监控alpha是否衰减/失效
        """
        monitoring = {
            'rolling_ic_30d': None,
            'rolling_ic_60d': None,
            'ic_vs_backtest': None,
            'turnover_deviation': None,
            'factor_exposure_drift': None,
            'drawdown_alert': None,
        }

        # 实盘IC vs 回测IC
        backtest_ic = self.config.get('expected_ic', 0.03)
        recent_ic = self._compute_recent_ic(live_alpha, window=60)

        if recent_ic < backtest_ic * 0.5:
            monitoring['alert'] = 'WARNING: IC下降超过50%'
        elif recent_ic < 0:
            monitoring['alert'] = 'CRITICAL: IC转负'

        return monitoring

    def _standardize_signal(self, raw_signal: pd.DataFrame) -> pd.DataFrame:
        """信号标准化流程"""
        signal = raw_signal.copy()

        # 1. 去极值 (MAD方法)
        median = signal.median(axis=1)
        mad = (signal.sub(median, axis=0)).abs().median(axis=1)
        upper = median + 5 * 1.4826 * mad
        lower = median - 5 * 1.4826 * mad
        signal = signal.clip(lower=lower, upper=upper, axis=0)

        # 2. 缺失值填充
        signal = signal.fillna(0)

        # 3. 截面排名
        signal = signal.rank(axis=1, pct=True) - 0.5

        # 4. 行业中性化 (可选)
        if self.config.get('industry_neutralize', True):
            signal = self._industry_neutralize(signal)

        # 5. 市值中性化 (可选)
        if self.config.get('size_neutralize', True):
            signal = self._size_neutralize(signal)

        return signal

    def _print_evaluation_report(self, report: dict):
        """打印评估报告"""
        pp = report['predictive_power']
        pnl = report['pnl_metrics']
        to = report['turnover_analysis']
        score = report['overall_score']

        print("=" * 70)
        print("           ALPHA EVALUATION REPORT")
        print("=" * 70)

        print(f"\n📊 Predictive Power:")
        print(f"   Rank IC (mean):        {pp['rank_ic_mean']:.4f}")
        print(f"   Rank IC (std):         {pp['rank_ic_std']:.4f}")
        print(f"   Information Ratio:     {pp['information_ratio']:.4f}")
        print(f"   IC Positive Ratio:     {pp['ic_positive_ratio']:.2%}")
        print(f"   IC t-statistic:        {pp['ic_t_stat']:.2f}")

        print(f"\n💰 PnL Metrics (Long-Short):")
        print(f"   Annual Return:         {pnl['annual_return']:.2%}")
        print(f"   Annual Volatility:     {pnl['annual_volatility']:.2%}")
        print(f"   Sharpe Ratio:          {pnl['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown:          {pnl['max_drawdown']:.2%}")
        print(f"   Monthly Win Rate:      {pnl['monthly_win_rate']:.2%}")

        print(f"\n🔄 Turnover:")
        print(f"   Daily Turnover:        {to['avg_daily_turnover']:.2%}")
        print(f"   Annual Turnover:       {to['annual_turnover']:.1f}x")
        print(f"   Holding Period:        {to['implied_holding_period_days']:.1f} days")
        print(f"   Breakeven Cost:        {to['breakeven_cost_bps']:.1f} bps")

        for cost, metrics in to['cost_scenarios'].items():
            print(f"   Net Sharpe @{cost}:     {metrics['net_sharpe']:.2f}")

        print(f"\n🏆 Overall Score:          {score['overall_score']:.1f} / 10")
        print(f"   Pass Threshold:         {'✅ YES' if score['passed_threshold'] else '❌ NO'}")
        print("=" * 70)

    def _compute_raw_signal(self, data, idea):
        pass

    def _industry_neutralize(self, signal):
        pass

    def _size_neutralize(self, signal):
        pass

    def _compute_recent_ic(self, alpha, window):
        pass
```

---

## 十、各家基金风格对比

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          顶级量化基金 Alpha 框架对比                         │
├────────────┬──────────────────┬──────────────────┬──────────────────────────┤
│   维度      │   Two Sigma      │   AQR            │   WorldQuant            │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 哲学        │ ML/AI驱动        │ 学术因子驱动      │ 大规模算子搜索           │
│            │ 数据即alpha      │ 经济学直觉        │ 数量取胜                 │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ Alpha来源   │ 替代数据+NLP     │ Value/Mom/Quality│ 遗传编程搜索             │
│            │ 深度学习         │ 跨资产carry       │ 10,000+ alphas          │
│            │ 强化学习         │ BAB/低风险异常    │ 全球覆盖70+国家          │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 模型偏好    │ GBM, Transformer │ OLS, Fama-MacBeth│ 表达式树/GP              │
│            │ NN, Random Forest│ 简洁线性模型      │ 简单快速评估             │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 数据投入    │ $$$$ (极高)      │ $$ (中等)        │ $$$ (较高)              │
│            │ 卫星/地理/NLP    │ 传统金融数据      │ 全球价量数据             │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 持仓期     │ 天 ~ 周          │ 周 ~ 月          │ 天 ~ 周                 │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 容量偏好    │ 中等 ($5-50B)    │ 大 ($10-100B+)   │ 小-中 ($1-10B)          │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 过拟合防范  │ 正则化/Dropout   │ 样本外/跨市场    │ IS/OOS严格分离           │
│            │ 时序交叉验证     │ 长历史回测        │ 衰减检测                 │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 基础设施    │ 自建大数据平台    │ 学术+工程混合    │ WebSim在线平台           │
│            │ Hadoop/Spark     │                  │ 众包研究员               │
├────────────┼──────────────────┼──────────────────┼──────────────────────────┤
│ 人才结构    │ CS/ML PhD        │ Finance PhD      │ 全球分布式               │
│            │ 工程师为主       │ 学术研究员为主    │ 研究顾问模式             │
└────────────┴──────────────────┴──────────────────┴──────────────────────────┘
```

---

## 十一、关键防过拟合策略

```python
class OverfittingPrevention:
    """
    防过拟合策略总结

    量化投资中过拟合是最大的敌人
    所有顶级基金都在这上面投入大量精力
    """

    STRATEGIES = {
        "1. 样本外测试": {
            "方法": "严格的IS/OOS分离，永远不在OOS上调参",
            "比例": "IS:60% / Validation:20% / OOS:20%",
            "关键": "OOS只看一次，看了就不能改",
        },

        "2. 多重检验校正": {
            "方法": "Bonferroni / BH / Deflated Sharpe Ratio",
            "原因": "搜索1000个alpha，有些会偶然显著",
            "公式": "Deflated SR = SR * sqrt(T) / sqrt(1 + skew*SR/6 + kurt*SR²/24)",
            "参考": "Harvey, Liu, Zhu (2016) '...and the Cross-Section of Expected Returns'",
            "threshold": "t-stat > 3.0 (而非传统的2.0)",
        },

        "3. 交叉验证": {
            "方法": "Purged Walk-Forward CV",
            "关键": "purge gap防止数据泄露",
            "实现": "TimeSeriesSplit + embargo period",
        },

        "4. 跨市场/跨时段验证": {
            "方法": "在美国发现的因子，在欧洲/亚洲也必须有效",
            "AQR": "所有因子必须跨40+国家有效",
            "时间": "分子样本期前后段分别验证",
        },

        "5. 经济直觉": {
            "方法": "Alpha必须有合理的经济学解释",
            "AQR座右铭": "If you can't explain it to your grandmother, don't trade it",
            "排除": "纯数据挖掘无解释的pattern",
        },

        "6. 复杂度惩罚": {
            "方法": "奥卡姆剃刀 — 简单模型优先",
            "实现": "alpha表达式树深度限制, L1/L2正则化",
            "WorldQuant": "偏好短表达式 (< 10个节点)",
        },

        "7. 参数稳定性": {
            "方法": "参数微小变化不应导致结果剧变",
            "实现": "参数敏感性分析 (±20%扰动)",
        },

        "8. Deflated Sharpe Ratio": {
            "公式": """
            给定: 
            - 尝试了 N 个策略
            - 最佳策略的样本 Sharpe = SR*
            - 回测长度 T

            P(SR* > 0 | N trials) 远大于 P(SR > 0 | 1 trial)

            DSR = SR* - SR_haircut(N, T, skewness, kurtosis)
            """,
        },
    }


def deflated_sharpe_ratio(observed_sr: float,
                           n_trials: int,
                           t_observations: int,
                           skewness: float = 0,
                           kurtosis: float = 3) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2014)

    调整了多重检验后的Sharpe Ratio

    Args:
        observed_sr: 观测到的样本Sharpe ratio (年化)
        n_trials: 尝试的策略/alpha数量
        t_observations: 独立观测数量
        skewness: 收益偏度
        kurtosis: 收益峰度

    Returns:
        p_value: 观测SR在调整后显著的概率
    """
    from scipy.stats import norm

    # 预期在N次随机试验中的最大SR (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772156649
    expected_max_sr = (
        (1 - euler_mascheroni) * norm.ppf(1 - 1 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # SR的标准误
    sr_std = np.sqrt(
        (1 - skewness * observed_sr + (kurtosis - 1) / 4 * observed_sr ** 2) / t_observations
    )

    # DSR的z统计量
    z = (observed_sr - expected_max_sr) / sr_std

    # p值
    p_value = norm.cdf(z)

    return p_value
```

---

这个框架覆盖了从**数据采集** → **因子挖掘** → **评估筛选** → **组合构建** → **回测验证** → **实盘监控**的完整链路，每个环节都体现了顶级量化基金的核心方法论和工程实践。