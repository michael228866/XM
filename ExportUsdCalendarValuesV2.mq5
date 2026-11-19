// Read-only export of USD economic-calendar values for the H-024 data qualification (STRATEGY_RESEARCH_2026_11.md 8.4).
// Based on ExportUsdCalendarValues.mq5; adds value_id/period/revision/revised_previous/export_time and never overwrites.
// No trading functions. Times are trade-server epoch seconds, NOT converted to UTC.
// Values: MT5 longs / 10^6 written with 6 decimals; LONG_MIN (missing) -> empty cell. 0 stays "0.000000".
// Output: Common\Files\usd_calendar_values_v2_<server time>.csv, UTF-8, written to .part first and renamed only when
// every read and every write succeeded. Any failure leaves the .part (do not use it) and shows no success message.
#property script_show_inputs
input datetime InpFrom = D'2014.01.01';
input datetime InpTo   = 0;   // 0 = now (released values only); a future date also exports upcoming forecasts

string Val(long v) { return v == LONG_MIN ? "" : DoubleToString(v / 1000000.0, 6); }

void Fail(int handle, string part, string what, int total)
{
   int err = GetLastError();
   FileClose(handle);
   Alert("Export FAILED (", what, ", error ", err, "); kept ", part, " - do not use it. Rows written: ", total);
}

void OnStart()
{
   datetime export_time = TimeTradeServer();
   datetime to = InpTo == 0 ? export_time : InpTo;
   if(InpFrom <= 0 || to <= 0 || InpFrom >= to)
   { Alert("Invalid range: InpFrom ", TimeToString(InpFrom), " must be earlier than end ", TimeToString(to)); return; }
   MqlDateTime t; TimeToStruct(export_time, t);
   string name = StringFormat("usd_calendar_values_v2_%04d%02d%02d_%02d%02d%02d.csv", t.year, t.mon, t.day, t.hour, t.min, t.sec);
   string part = name + ".part";
   if(FileIsExist(name, FILE_COMMON) || FileIsExist(part, FILE_COMMON))
   { Alert("Refusing to overwrite existing ", name, " (or .part); run again later."); return; }

   ResetLastError();
   int handle = FileOpen(part, FILE_WRITE | FILE_CSV | FILE_ANSI | FILE_COMMON, ',', CP_UTF8);
   if(handle == INVALID_HANDLE) { Alert("FileOpen failed: ", GetLastError()); return; }
   if(FileWrite(handle, "value_id", "event_id", "importance", "scheduled_time", "period", "revision",
                "actual", "forecast", "previous", "revised_previous", "impact_type", "export_time", "name") == 0)
   { Fail(handle, part, "header write", 0); return; }

   int total = 0;
   for(datetime from = InpFrom; from < to; from += 365 * 86400)
   {
      MqlCalendarValue values[];
      datetime chunk_end = (datetime)MathMin(from + 365 * 86400, to);
      ResetLastError();
      if(CalendarValueHistory(values, from, chunk_end, NULL, "USD") < 0)
      { Fail(handle, part, "CalendarValueHistory " + TimeToString(from), total); return; }
      for(int i = 0; i < ArraySize(values); i++)
      {
         if(values[i].time >= chunk_end) continue;   // chunk boundaries are inclusive; avoid duplicates
         MqlCalendarEvent ev;
         ResetLastError();
         if(!CalendarEventById(values[i].event_id, ev))
         { Fail(handle, part, "CalendarEventById " + (string)values[i].event_id, total); return; }
         string ev_name = ev.name;
         StringReplace(ev_name, ",", ";");             // FILE_CSV does not quote
         ResetLastError();
         if(FileWrite(handle, (long)values[i].id, (long)values[i].event_id, (int)ev.importance,
                      (long)values[i].time, (long)values[i].period, (int)values[i].revision,
                      Val(values[i].actual_value), Val(values[i].forecast_value),
                      Val(values[i].prev_value), Val(values[i].revised_prev_value),
                      (int)values[i].impact_type, (long)export_time, ev_name) == 0)
         { Fail(handle, part, "row write", total); return; }
         total++;   // counted only after a confirmed write
      }
   }
   ResetLastError();
   FileFlush(handle);   // flush buffers before close; not a guarantee against power or disk failure
   if(GetLastError() != 0) { Fail(handle, part, "flush", total); return; }
   FileClose(handle);
   ResetLastError();
   if(!FileMove(part, FILE_COMMON, name, FILE_COMMON))   // no FILE_REWRITE: fails rather than overwrite
   { Alert("Rename failed (", GetLastError(), "); data kept in ", part, " - do not use it."); return; }
   if(total == 0)
      Alert("Export finished with 0 rows -> Common\\Files\\", name, " (nothing to analyse)");
   else
      Alert("USD calendar values export done: ", total, " rows -> Common\\Files\\", name);
}
