#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import numpy as np

#Pandas version 0.22.0
#Bokeh version 0.12.10
#Numpy version 1.12.1

from bokeh.io import output_file, show,curdoc
from bokeh.models import Quad
from bokeh.models import NumeralTickFormatter
from bokeh.models import PrintfTickFormatter
from bokeh.layouts import row, layout,widgetbox
from bokeh.models.widgets import Select,MultiSelect, Button, CheckboxGroup, Slider, DateRangeSlider, Paragraph, Div
from bokeh.plotting import ColumnDataSource,Figure,reset_output,gridplot
from bokeh.models import ranges, LabelSet
from bokeh.models.glyphs import Line
from os.path import join, dirname
import datetime
from datetime import date
from bokeh.palettes import Spectral11
from bokeh.transform import cumsum
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import formatters
import math
from bokeh.embed import components
from bokeh.models import ColorBar
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap


from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models import ColumnDataSource, DataRange1d, Select, FuncTickFormatter
from bokeh.palettes import Blues4
from bokeh.plotting import figure
from bokeh.models.glyphs import Line
from bokeh.transform import dodge

from bokeh.document import Document
from bokeh.models import ColumnDataSource, Plot, LinearAxis, Grid, Circle, HoverTool, BoxSelectTool, LassoSelectTool, ResetTool
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, NumberFormatter, StringEditor, IntEditor, NumberEditor, SelectEditor
from bokeh.models.tickers import SingleIntervalTicker
from bokeh.models.axes import Axis
from bokeh.models.layouts import Column
from bokeh.embed import file_html
from bokeh.resources import INLINE
from bokeh.util.browser import view
from math import pi
from datetime import datetime
from bokeh.models import CustomJS


from datetime import datetime
from pathlib import Path


from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter, Paragraph
from bokeh.layouts import column

import sys



from pathlib import Path

from memory_profiler import profile







# In[67]:


ep = pd.read_csv("C:/Users/lsahi/Documents/Lakamana_GMU_Sem3/DAEN690/ADMISSIONS.csv")
ep = ep[pd.notnull(ep['DIAGNOSIS'])]
ep = ep[pd.notnull(ep['DISCHARGE_LOCATION'])]
ep = ep[pd.notnull(ep['MARITAL_STATUS'])]
ep['HOSPITAL_EXPIRE_FLAG'] = ep['HOSPITAL_EXPIRE_FLAG'].astype(str)
ep['HAS_CHARTEVENTS_DATA'] = ep['HAS_CHARTEVENTS_DATA'].astype(str)


# In[68]:


# noteevents = pd.read_csv("C:/Users/lsahi/Documents/Lakamana_GMU_Sem3/DAEN690/NOTEEVENTS.csv", low_memory=False)


# In[69]:


ep.index


# In[70]:


#category = admission type
adm_type = (sorted(ep['ADMISSION_TYPE'].astype(str).unique()))
adm_none = ""
adm_type

#marital status = naming
mar_stat = sorted((ep['MARITAL_STATUS'].unique()))
mar_stat

#fractures
flag = str(ep['HOSPITAL_EXPIRE_FLAG'])
fractures = (sorted(ep['HOSPITAL_EXPIRE_FLAG'].astype(str).unique()))
fractures

ethnicity = (sorted(ep['ETHNICITY'].unique()))
ethnicity

#drops = has chartevents data
drops = (sorted(ep['HAS_CHARTEVENTS_DATA'].astype(str).unique()))
drops

insurance = (sorted(ep['INSURANCE'].unique()))
insurance

#New
diagnosis = (sorted(ep['DIAGNOSIS'].unique()))

#New
disch_loc = (sorted(ep['DISCHARGE_LOCATION'].unique()))
disch_loc


# In[71]:


adm_type


# In[72]:


ep['ADMITTIME'] = pd.to_datetime(ep['ADMITTIME'])
ep['DISCHTIME'] = pd.to_datetime(ep['DISCHTIME'])


# In[73]:


ep1 = ep.groupby(by = ['ADMISSION_TYPE', 'HOSPITAL_EXPIRE_FLAG', 'HAS_CHARTEVENTS_DATA', 'MARITAL_STATUS'])
ep1 = ep1.count()
ep1 = ep1.iloc[:,0]
ep1 = ep1.reset_index()


# In[74]:


#start = start_adm, end = end_adm
start_adm = min(ep['ADMITTIME'])
end_adm = max(ep['ADMITTIME'])

start_disch = min(ep['DISCHTIME'])
end_disch = max(ep['DISCHTIME'])


# In[75]:

@profile(precision=4)
def update_plot(attrname, old, new):

    fracture = fracture_select.value
    adm_type = [category_selection.labels[i] for i in
                        category_selection.active]
    drop = drop_select.value
#     date12     = date_range_slider.value_as_date
#     start_adm, end_adm = date12

    #Considering dataset with the state, fractures, pgp and surgeon restriction
    ep1 = get_dataset(df, epa, ep_pie, fracture, adm_type, drop)

    source.data = ep1.data


# In[76]:


adm_type
adm = pd.DataFrame(adm_type)
adm.dtypes


# In[77]:


fracture = '0'
drop = '0'


# In[78]:


#Loading the dataset
@profile(precision=4)
def get_dataset(ep, epa, ep_pie, fracture1 ,adm_type_1a, drop1):
    df = ep[(ep.HOSPITAL_EXPIRE_FLAG == fracture1) & (ep.ADMISSION_TYPE.isin(adm_type_1a)) & (ep.HAS_CHARTEVENTS_DATA == drop1)].copy()
#     df_a = epa[ (epa.HOSPITAL_EXPIRE_FLAG == fracture1) & (epa.ADMISSION_TYPE.isin(category1_a)) & (epa.HAS_CHARTEVENTS_DATA == drop1) & ((epa.ADMITTIME) >= datestart) & ((epa.ADMITTIME) <= dateend)].copy()
#     df_pie = ep_pie[ (ep_pie.HOSPITAL_EXPIRE_FLAG == fracture1) & (ep_pie.ADMISSION_TYPE.isin(category1_a)) & (ep_pie.HAS_CHARTEVENTS_DATA == drop1) & ((ep_pie.ADMITTIME) >= datestart) & ((ep_pie.ADMITTIME) <= dateend)].copy()

    #Remove so as to update the dataframe
    del df['HOSPITAL_EXPIRE_FLAG']
    del df['ADMISSION_TYPE']
    del df['HAS_CHARTEVENTS_DATA']

    return ColumnDataSource(data=df)


# In[79]:

@profile(precision=4)
def make_plot(source, title):

    y_label = "Ethnicity"
    x_label = "Index"
    title = "Ethnicity Survey"


#     mapper = linear_cmap( field_name = 'tot_std_allowed',palette=Spectral6 ,low=min(source.data['tot_std_allowed']) ,high=max(source.data['tot_std_allowed']))

    plot = figure(plot_width=1000, plot_height= 500,
            x_axis_label = x_label,
            y_axis_label = y_label,
            title=title,
            x_minor_ticks=4,
            x_range = mar_stat,
            tools="pan, box_select, lasso_select, reset, save ", active_drag="lasso_select")

#     labels = LabelSet(x='ROW_ID', y='HADM_ID', text='tot_std_allowed', level='glyph', text_font_size= "8pt",
#             x_offset= -26, y_offset= 0, source=source, render_mode='canvas')

#     cty = plot.vbar(source=source ,x= 'pathways', top='tot_std_allowed', width = 0.4,line_color = mapper ,color = mapper)
    cty2 = plot.line(x = 'MARITAL_STATUS', y= 'index', line_width = 3, source = source, color = 'orange')
    plot.xaxis.major_label_orientation = math.pi/6

#     plot.add_layout(labels)
    # plot.yaxis.formatter.use_scientific = False
    plot.xaxis.axis_label_text_font_size = "12pt"
    plot.xaxis.major_label_text_font_size = "10pt"
    plot.xaxis.axis_label_text_font = "calibiri"

    plot.title.align = "center"

    plot.yaxis.axis_label_text_font_size = "12pt"
    plot.yaxis.major_label_text_font_size = "10pt"
    plot.yaxis.axis_label_text_font = "calibiri"
#     plot.yaxis[0].formatter = NumeralTickFormatter(format= '($ 0,0 a)')

    #Outline
    plot.outline_line_color = "navy"

    plot.ygrid.band_fill_alpha = 0.1
    plot.ygrid.band_fill_color = "navy"

    # plot.title.text_color = "orange"
    plot.title.text_font_size = "16px"
    # plot.title.background_fill_color = "#aaaaee"
    plot.border_fill_color = "whitesmoke"

    return plot


# In[80]:

@profile(precision=4)
def update():
    category_selection.active = list(range(len(adm_type)))

@profile(precision=4)
def update_clear():
    category_selection.active = list(range(len(adm_none)))


# In[81]:


#Providing select widget for defining filters
fracture_select = Select(value = '0', title = 'Fracture Flag', options= sorted(fractures))
drop_select = Select(value = '0', title = 'Drop Episode', options = sorted(drops))
category_selection = CheckboxGroup(labels= adm_type,
                                  active = [0,1,2,3])
select_all = Button(label="Select all")
clear_all = Button(label = "Clear all")
text_title_cat = Div( text = """ <b> <u> Category List <b> <u> """, width = 200)
text_title_dat = Div( text = """ <b> <u> Data Table for Utilization <b> <u> """, width = 200)
text_title_dat_e = Div( text = """ <b> <u> Data Table for Episode Count <b> <u> """, width = 200)
# date_range_slider = DateRangeSlider(title="", start = datetime(2100,6,7), end=datetime(2210, 8, 17), value=(datetime(2100, 6, 7), datetime(2210, 8, 17)), step= 1)

#variables that enables to update the plot
fracture_select.on_change('value', update_plot)
drop_select.on_change('value', update_plot)
category_selection.on_change('active', update_plot)
select_all.on_click(update)
clear_all.on_click(update_clear)
# date_range_slider.on_change('value', update_plot)


# In[82]:


adm_type


# In[83]:


#Calling all functions
ep = ep1
df = ep1
epa = ep1
ep_pie = ep1
source = get_dataset(df, epa, ep_pie, fracture, adm_type, drop)


plot = make_plot(source, "Utilization for each pathway" )
controls = widgetbox(fracture_select, drop_select, text_title_cat, category_selection, select_all, clear_all, width = 250)


# In[84]:


plot = layout([[controls, plot]])
show(plot)
curdoc().add_root(plot)


# In[ ]:
