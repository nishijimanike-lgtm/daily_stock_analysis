import os
import sys
from datetime import datetime
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

# Import our system components
from data_provider import DataFetcherManager, EfinanceFetcher, AkshareFetcher, TushareFetcher, PytdxFetcher, BaostockFetcher, YfinanceFetcher
from src.storage import get_db
from sqlalchemy import text

def test_db():
    print("\n--- [测试 1: 数据库连接] ---")
    try:
        db = get_db()
        session = db.get_session()
        result = session.execute(text("SELECT 1")).fetchone()
        if result[0] == 1:
            print("✓ 数据库连接正常")
        
        # Check table exist
        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_daily'")).fetchone()
        if result:
            print("✓ 'stock_daily' 数据表已创建")
            count = session.execute(text("SELECT COUNT(*) FROM stock_daily")).fetchone()[0]
            print(f"  当前记录数: {count}")
        else:
            print("⚠ 'stock_daily' 数据表尚未创建，将首次运行分析时自动创建")
        session.close()
    except Exception as e:
        print(f"✗ 数据库连接错误: {e}")

def test_daily_fetchers(stock_code="600519"):
    print(f"\n--- [测试 2: 日线数据接口 (测试股票: {stock_code})] ---")
    
    fetchers_classes = [
        EfinanceFetcher,
        AkshareFetcher,
        BaostockFetcher,
        YfinanceFetcher
    ]
    
    for f_class in fetchers_classes:
        try:
            f = f_class()
            print(f"尝试 [{f.name}]...")
            df = f.get_daily_data(stock_code, days=5)
            if not df.empty:
                print(f"  ✓ [{f.name}] 联通正常 (获取到 {len(df)} 条数据)")
            else:
                print(f"  ⚠ [{f.name}] 返回空数据")
        except Exception as e:
            print(f"  ✗ [{f_class.__name__ if hasattr(f_class, '__name__') else 'Fetcher'}] 错误: {e}")

def test_realtime_sources(stock_code="600519"):
    print(f"\n--- [测试 3: 实时行情接口 (测试股票: {stock_code})] ---")
    
    manager = DataFetcherManager()
    
    # Injected test for realtime sources
    from src.config import get_config
    config = get_config()
    
    print(f"当前配置优先级: {config.realtime_source_priority}")
    
    try:
        quote = manager.get_realtime_quote(stock_code)
        if quote:
            print(f"✓ 实时行情获取成功 (主引擎自动切换)")
            print(f"  数据来源: {getattr(quote, 'source', 'Unknown')}")
            print(f"  当前价格: {quote.price}")
            print(f"  涨跌幅: {quote.change_pct}%")
            print(f"  成交量: {quote.volume}")
            print(f"  基本数据齐全: {'是' if quote.has_basic_data() else '否'}")
        else:
            print("✗ 无法获取实时行情数据")
    except Exception as e:
        print(f"✗ 实时行情异常: {e}")

if __name__ == "__main__":
    print(f"🚀 启动数据接口全面联通性测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    test_db()
    test_daily_fetchers()
    test_realtime_sources()
    print("\n--- 测试全部完成 ---")
