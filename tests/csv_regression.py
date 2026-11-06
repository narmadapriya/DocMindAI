import importlib.util,csv
from pathlib import Path
file=Path(__file__).resolve().parents[1]/'backend/app/services/summary_service.py'
spec=importlib.util.spec_from_file_location('candidate', file); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=Path('/mnt/data/apex_motors_monthly_sales_2025.csv')
if source.exists():
    with source.open(newline='') as handle:
        rows=list(csv.DictReader(handle))
else:
    rows=[dict(Month=month,Year='2025',**{'Units Sold':str(units),'Revenue (USD)':str(rev)}) for month,units,rev in
          [('Jan',1050,38640000),('Feb',1130,41810000),('Mar',1240,46128000),('Apr',1260,47124000),('May',1290,48504000),('Jun',1340,50652000),('Jul',1330,50540000),('Aug',1380,52716000),('Sep',1440,55296000),('Oct',1480,57128000)]]
content='CSV table\nColumns: Month, Year, Units Sold, Revenue (USD)\n'+'\n'.join('Row '+str(i+2)+': '+ '; '.join(f'{k}: {v}' for k,v in r.items()) for i,r in enumerate(rows))
evidence=[{'chunk_id':'c1','content':content,'metadata':{'filename':'apex_motors_monthly_sales_2025.csv','file_type':'csv'}}]
for scope in ('document','executive'):
 s=m.SummaryService._source_grounded_tabular_presentation(evidence=evidence,scope=scope)
 print(scope,repr(s[:1250]) if s else 'NO SUMMARY')
 assert s and '12,940' in s and '488,538,000' in s
 assert '18,488,000' in s and '50,652,000' in s and '50,540,000' in s
print('PASS: CSV totals and computed trend are evidence-derived')
