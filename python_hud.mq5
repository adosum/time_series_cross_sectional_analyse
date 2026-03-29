//+------------------------------------------------------------------+
//|                                                   Python_HUD.mq5 |
//+------------------------------------------------------------------+
#property indicator_chart_window
#property indicator_plots 0

// 初始化函数：设置定时器
int OnInit()
  {
   // 每 1 秒钟触发一次读取
   EventSetTimer(1); 
   return(INIT_SUCCEEDED);
  }

// 清理函数：关闭时抹掉左上角的字
void OnDeinit(const int reason)
  {
   EventKillTimer();
   Comment(""); 
  }

// 核心心跳：读取 Python 留下的文件并显示
void OnTimer()
  {
   // 尝试打开文件 (开启共享读取权限，防止和 Python 冲突)
   int file_handle = FileOpen("python_dashboard.txt", FILE_READ|FILE_TXT|FILE_ANSI|FILE_SHARE_READ);
   
   if(file_handle != INVALID_HANDLE)
     {
      string display_text = "";
      // 逐行读取文件内容
      while(!FileIsEnding(file_handle))
        {
         display_text += FileReadString(file_handle) + "\n";
        }
      FileClose(file_handle);
      
      // 神奇的 Comment 函数，直接把文字贴在图表左上角！
      Comment(display_text); 
     }
  }

// 必须保留的事件，但不做任何事
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   return(rates_total);
  } 