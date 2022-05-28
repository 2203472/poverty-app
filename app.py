import math

import pickle

import flask
import pandas as pd
from scipy.stats import stats
from sklearn import preprocessing
import seaborn as sns

from flask import request, url_for, redirect, render_template
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Use pickle to load in the pre-trained model.
with open(f'model/povery-prediction-lightgbm.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__,
                  static_folder='static',
                  template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    # 2015 Household Expenditure
    img = BytesIO()
    data = pd.read_csv('datasets/fies-cleaned.csv')
    cg_font = {"fontname": "Century Gothic"}
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(15, 10))
    ax = sns.barplot(x="region", y="total_expenditure", palette="pastel", edgecolor=".6", hue="class", data=data)
    ax.set_xticklabels(ax.get_xticklabels(), fontname="Century Gothic")
    plt.xticks(rotation=70)

    ax.set_xlabel("Region")
    ax.set_ylabel("Total Expenditure")
    plt.title('2015 Average Household Expenditure', cg_font, fontsize=20)

    plt.tight_layout()

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    expenditure_rgn = base64.b64encode(img.getvalue()).decode('utf8')

    # Poor Household Expenditure
    poor_data = data.loc[data['class'] == 'Poor']
    label = ['bread_cereals_expense', 'rice_expense', 'meat_expense', 'fish_marine_expense',
             'fruit_expense', 'veg_expense', 'restohotel_expense', 'alcohol_expense',
             'tobacco_expense', 'wear_expense', 'houseing_water_expense', 'imp_houserental',
             'medcare_expense', 'transpo_expense', 'commu_expense', 'edu_expense', 'misc_expense',
             'specialoc_expense', 'farming_garden_expense']

    d = {"Bread & Cereal Expenses": 'bread_cereals_expense', "Rice Expenses": 'rice_expense',
         "Meat Expenses": 'meat_expense', "Fish Expenses": 'fish_marine_expense',
         "Fruit Expenses": 'fruit_expense', "Vegetable Expenses": 'veg_expense',
         "Resto and Hotel Expenses": "restohotel_expense", "Alcohol Expenses": 'alcohol_expense',
         "Tobacco Expenses": 'tobacco_expense', "Clothin Expenses": 'wear_expense',
         "House & Water Expenses": 'houseing_water_expense', "Imputed House Rental Value": 'imp_houserental',
         "Medical Expenses": 'medcare_expense', "Transportation Expenses": 'transpo_expense',
         "Communication Expenses": 'commu_expense', "Education Expenses": "edu_expense",
         "Miscellaneous Expenses": "misc_expense", "Occasion Expenses": "specialoc_expense",
         "Farming & Garden Expenses": "farming_garden_expense"}

    poor_data_mydict = {}

    temp = dict((v, k) for k, v in d.items())

    for x in label:
        poor_data_mydict[x] = poor_data[x].mean()

    poor_data_sorted_tuple = sorted(poor_data_mydict.items(), key=lambda x: x[1], reverse=True)
    poor_data_sorted_dict = {k: v for k, v in poor_data_sorted_tuple}

    clean_dict = {temp[k]: v for k, v in poor_data_sorted_dict.items()}

    keys = list(clean_dict.keys())
    values = list(clean_dict.values())

    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=values, y=keys, palette="pastel", edgecolor=".6")
    ax.set_xlabel("Average Expenditures", cg_font)
    plt.title('Average Poor Household Expenditure', cg_font, fontsize=20)
    ax.bar_label(ax.containers[0], padding=3, fontname="Century Gothic")

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pr_expenditure_rgn = base64.b64encode(img.getvalue()).decode('utf8')

    with open(f'datasets/raw-features.pkl', 'rb') as f:
        df = pickle.load(f)

    if flask.request.method == 'GET':
        return flask.render_template('main.html', expenditure_rgn=expenditure_rgn,
                                     pr_expenditure_rgn=pr_expenditure_rgn)

    if flask.request.method == 'POST':
        if request.form['btn'] == 'predict':
            household_income = float(flask.request.form['houseincome'])
            bread_cereals_expense = float(flask.request.form['breadcerealexp'])
            rice_expense = float(flask.request.form['riceexp'])
            meat_expense = float(flask.request.form['meatexp'])
            fish_marine_expense = float(flask.request.form['fishmarineexp'])
            fruit_expense = float(flask.request.form['fruitexp'])
            veg_expense = float(flask.request.form['veggieexp'])
            restohotel_expense = float(flask.request.form['restohotelexp'])
            wear_expense = float(flask.request.form['wearexp'])
            housing_water_expense = float(flask.request.form['housingwaterexp'])
            imp_houserental = float(flask.request.form['imphouserental'])
            transpo_expense = float(flask.request.form['transpoexp'])
            commu_expense = float(flask.request.form['commuexp'])
            edu_expense = float(flask.request.form['eduexp'])
            misc_expense = float(flask.request.form['miscexp'])
            farming_garden_expense = float(flask.request.form['farmgardenexp'])

            food_expense = float(bread_cereals_expense) + float(rice_expense) + float(meat_expense) + float(
                fish_marine_expense) + float(fruit_expense) + float(veg_expense)

            df.loc[len(df)] = [household_income, food_expense, bread_cereals_expense, meat_expense, rice_expense,
                               fruit_expense, fish_marine_expense, veg_expense, restohotel_expense, wear_expense,
                               housing_water_expense, imp_houserental, transpo_expense, commu_expense, edu_expense,
                               misc_expense, farming_garden_expense, 5]

            # Normalization
            x = df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            df = pd.DataFrame(min_max_scaler.fit_transform(x), index=df.index, columns=df.columns)
            input_variables = df.tail(1)

            prediction = model.predict(input_variables, predict_disable_shape_check=True)[0]

            if prediction >= 0.5:
                printpredict = "Poor"
            else:
                printpredict = "Not Poor"

            return flask.render_template('main.html', result=printpredict)

        if request.form['btn'] == 'Plot':
            region = request.form["region"]
            return redirect(url_for("plot", rgn=region))


@app.route('/<rgn>')
def plot(rgn):
    df = pd.read_csv('datasets/fies-cleaned.csv')
    colors = ['#DF9D9E', '#BF899C', '#C79272', '#987896', '#9E8C54', '#716987',
              '#67864C', '#4C5972', '#117D5B', '#2F4858', '#006F74']
    cg_font = {"fontname": "Century Gothic"}
    region_value = rgn
    img = BytesIO()

    # Poor Average Household income
    poor_data = df.loc[(df['class'].isin(["Poor"])) & (df['region'].isin([region_value]))]
    rgn_poor_household_expenditures = poor_data['total_expenditure']
    poor_household_expenditures = "{:.2f}".format(rgn_poor_household_expenditures.mean())

    # Not Poor Average Household income
    not_poor_data = df.loc[(df['class'].isin(["Not Poor"])) & (df['region'].isin([region_value]))]
    rgn_not_poor_household_expenditures = not_poor_data['total_expenditure']
    not_poor_household_expenditures = "{:.2f}".format(rgn_not_poor_household_expenditures.mean())

    # Poor Average Expenditure based on region
    label = ['bread_cereals_expense', 'rice_expense', 'meat_expense', 'fish_marine_expense',
             'fruit_expense', 'veg_expense', 'restohotel_expense', 'alcohol_expense',
             'tobacco_expense', 'wear_expense', 'houseing_water_expense', 'imp_houserental',
             'medcare_expense', 'transpo_expense', 'commu_expense', 'edu_expense', 'misc_expense',
             'specialoc_expense', 'farming_garden_expense']

    d = {"Bread & Cereal Expenses": 'bread_cereals_expense', "Rice Expenses": 'rice_expense',
         "Meat Expenses": 'meat_expense', "Fish Expenses": 'fish_marine_expense',
         "Fruit Expenses": 'fruit_expense', "Vegetable Expenses": 'veg_expense',
         "Resto and Hotel Expenses": "restohotel_expense", "Alcohol Expenses": 'alcohol_expense',
         "Tobacco Expenses": 'tobacco_expense', "Clothin Expenses": 'wear_expense',
         "House & Water Expenses": 'houseing_water_expense', "Imputed House Rental Value": 'imp_houserental',
         "Medical Expenses": 'medcare_expense', "Transportation Expenses": 'transpo_expense',
         "Communication Expenses": 'commu_expense', "Education Expenses": "edu_expense",
         "Miscellaneous Expenses": "misc_expense", "Occasion Expenses": "specialoc_expense",
         "Farming & Garden Expenses": "farming_garden_expense"}

    poor_data_mydict = {}

    temp = dict((v, k) for k, v in d.items())

    for x in label:
        poor_data_mydict[x] = poor_data[x].mean()

    poor_data_sorted_tuple = sorted(poor_data_mydict.items(), key=lambda x: x[1], reverse=True)
    poor_data_sorted_dict = {k: v for k, v in poor_data_sorted_tuple}

    clean_dict = {temp[k]: v for k, v in poor_data_sorted_dict.items()}

    keys = list(clean_dict.keys())
    values = list(clean_dict.values())

    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x=values, y=keys, palette="pastel", edgecolor=".6")
    ax.set_xlabel("Average Expenditures", cg_font)
    plt.title('Average Poor Household Expenditure', cg_font, fontsize=20)
    ax.bar_label(ax.containers[0], padding=3, fontname="Century Gothic")

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pr_expenditure_rgn = base64.b64encode(img.getvalue()).decode('utf8')

    #  Poor Occupation count per region
    poor_occupation_count = poor_data['head_occupation'].value_counts()
    poor_occupation_count = poor_occupation_count[:10, ]

    plt.figure(figsize=(15, 7))
    ax1 = sns.barplot(x=poor_occupation_count.index, y=poor_occupation_count.values, palette="pastel",
                      edgecolor=".6", hue=poor_occupation_count.index, dodge=False)
    plt.ylabel('Occurrence', cg_font)
    plt.title('Top 10 Poor Household Head Occupation', cg_font, fontsize=20)

    for container in ax1.containers:
        ax1.bar_label(container, padding=3, fontname="Century Gothic")

    plt.xticks([])

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    bar_poor_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Head Highest attainment for poor
    poor_data_counts = poor_data['head_highestgrade'].value_counts()
    poor_data_counts = poor_data_counts[:10, ]

    plt.figure(figsize=(10, 5))
    ax2 = sns.barplot(x=poor_data_counts.values, y=poor_data_counts.index, palette="pastel", edgecolor=".6")
    ax2.bar_label(ax2.containers[0], padding=3, fontname="Century Gothic")

    plt.xlabel("Occurence", cg_font)
    plt.title('Top 10 Poor Household Head Highest Attainment', cg_font, fontsize=20)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    bar_head_attainment_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Poor type of house
    poor_data_counts = poor_data['rooftype'].value_counts()

    plt.figure(figsize=(10, 5))
    ax3 = sns.barplot(x=poor_data_counts.values, y=poor_data_counts.index, palette="pastel", edgecolor=".6")
    ax3.set(ylabel=None)
    plt.xlabel("Occurence", cg_font)
    plt.title('Poor Household Type Of House', cg_font, fontsize=20)
    ax3.bar_label(ax3.containers[0], padding=3, fontname="Century Gothic")

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    bar_typ_house_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Main source of income for poor people
    src_count = poor_data['mainsrc'].value_counts()
    src_list = list(src_count.keys())

    src_status = []
    for key in src_count.keys():
        src_status.append(src_count[key])

    plt.pie(src_status, labels=src_list,
            shadow=True, startangle=180,
            autopct='%1.1f%%', colors=colors)

    plt.title('Main Source of Income',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_mainsrc_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Financial Status Section
    region_data = df.loc[df['region'].isin([region_value])]
    status_count = region_data['class'].value_counts()
    status_list = list(status_count.keys())

    values_status = []
    for key in status_count.keys():
        values_status.append(status_count[key])

    plt.pie(values_status, labels=status_list,
            shadow=True, startangle=180,
            autopct='%1.1f%%',
            colors=colors)

    plt.title('Household Financial Status',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_status_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Poor toilet facility
    region_data = df.loc[df['class'].str.contains('Poor') & df['region'].str.contains('CAR')]
    toilet_fac_count = region_data['toilet_facility'].value_counts()
    toilet_fac_count = toilet_fac_count[:7, ]
    toilet_fac_list = list(toilet_fac_count.keys())

    toilet_fac_status = []
    for key in toilet_fac_count.keys():
        toilet_fac_status.append(toilet_fac_count[key])

    plt.figure(figsize=(6, 5))

    plt.pie(toilet_fac_status, labels=toilet_fac_list,
            shadow=True, startangle=180, colors=colors,
            autopct='%1.1f%%')

    plt.title('Poor Toilet Facility',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_toilet_fac_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Poor Tenure
    region_data = df.loc[df['class'].str.contains('Poor') & df['region'].str.contains('CAR')]
    tenure_count = region_data['tenure'].value_counts()
    tenure_count = tenure_count[:5, ]
    tenure_list = list(tenure_count.keys())

    tenure_status = []
    for key in tenure_count.keys():
        tenure_status.append(tenure_count[key])

    plt.figure(figsize=(7, 7))

    plt.pie(tenure_status, labels=tenure_list,
            shadow=True, startangle=180, colors=colors,
            autopct='%1.1f%%')

    plt.title('Poor Tenure',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_tenure_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Poor Water supply
    region_data = df.loc[df['class'].str.contains('Poor') & df['region'].str.contains('CAR')]
    wtr_sply_count = region_data['water_supply'].value_counts()
    wtr_sply_count = wtr_sply_count[:10, ]
    wtr_sply_list = list(wtr_sply_count.keys())

    wtr_sply_status = []
    for key in wtr_sply_count.keys():
        wtr_sply_status.append(wtr_sply_count[key])

    plt.figure(figsize=(8, 8))

    plt.pie(wtr_sply_status, labels=wtr_sply_list,
            shadow=True, startangle=180, colors=colors,
            autopct='%1.1f%%')

    plt.title('Poor Water supply',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_wtr_sply_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Household Possessions
    poor_data = df.loc[(df['class'] == 'Poor') & (df['region'] == region_value)]
    not_poor_data = df.loc[(df['class'] == 'Not Poor') & (df['region'] == region_value)]

    label = ['no_television', 'no_cd_vcd_dvd', 'no_component_stereo', 'no_ref', 'no_washingmachine',
             'no_airconditioner', 'no_car_jeep_van', 'no_landline_wireless', 'no_cp', 'no_pc', 'no_stovegas',
             'no_banca', 'no_motorcycle']

    d = {"Number of Television": 'no_television', "Number of VCD/DVD": 'no_cd_vcd_dvd',
         "Number of Component Stereo": 'no_component_stereo', "Number of Refrigerator": 'no_ref',
         "Number of Washing Machine": 'no_washingmachine', "Number of Air Conditioner": 'no_airconditioner',
         "Number of Car/Jeep/Van": 'no_car_jeep_van', "Number of Landline/Wireless Connection": 'no_landline_wireless',
         "Number of Phone": 'no_cp', "Number of Computer": 'no_pc',
         "Number of Gas Stove": 'no_stovegas', "Number of Bangka": 'no_banca',
         "Number of Motorcycle": 'no_motorcycle'}

    d = dict((v, k) for k, v in d.items())

    mydict_poor = {}

    for x in label:
        mydict_poor[x] = poor_data[x].sum()

    sorted_tuple_poor = sorted(mydict_poor.items(), key=lambda x: x[1])
    sorted_dict_poor = {k: v for k, v in sorted_tuple_poor}

    mydict_not_poor = {}

    for x in label:
        mydict_not_poor[x] = not_poor_data[x].sum()

    sorted_tuple_not_poor = sorted(mydict_not_poor.items(), key=lambda x: x[1])
    sorted_dict_not_poor = {k: v for k, v in sorted_tuple_not_poor}

    keys = list(sorted_dict_poor.keys())
    values_poor = list(sorted_dict_poor.values())
    values_not_poor = list(sorted_dict_not_poor.values())

    items_plt = pd.DataFrame({'Poor': values_poor, 'Not Poor': values_not_poor}, index=keys)
    items_plt = items_plt.rename(index=d)

    items_bar_plt = items_plt.plot.barh(figsize=(10, 6), width=1, color=['#F4BFBF', '#8CC0DE'])
    items_bar_plt.legend(loc='lower right')

    plt.title('Poor Household Possession', cg_font, fontsize=20)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    household_possessions_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Correlation on Household Possessions
    df = pd.read_csv('datasets/fies-cleaned.csv')
    df = df.loc[df['region'].str.contains(region_value)]

    to_drop = ['household_income', 'region', 'food_expense', 'mainsrc', 'agri_indicator',
               'bread_cereals_expense', 'rice_expense', 'meat_expense', 'fish_marine_expense',
               'fruit_expense', 'veg_expense', 'restohotel_expense', 'alcohol_expense',
               'tobacco_expense', 'wear_expense', 'houseing_water_expense', 'imp_houserental',
               'medcare_expense', 'transpo_expense', 'commu_expense', 'edu_expense', 'misc_expense',
               'specialoc_expense', 'farming_garden_expense', 'totalincome_entrep', 'head_sex', 'head_age',
               'head_marital', 'head_highestgrade', 'head_job_business', 'head_occupation', 'head_class_worker',
               'householdtype', 'total_fam_mem', 'memage_less5', 'memage_5-17', 'no_employedfam', 'bldghousetype',
               'rooftype', 'walltype', 'floor_area', 'house_age', 'no_bedrooms', 'tenure', 'toilet_facility',
               'electricity', 'water_supply']
    df.drop(to_drop, inplace=True, axis=1)

    classTypeDummies = pd.get_dummies(df['class'])

    df = pd.concat([df, classTypeDummies], axis='columns')

    to_drop = ['total_expenditure',
               'class',
               'Not Poor']
    df.drop(to_drop, inplace=True, axis=1)

    d_hp = {"Number of Television": 'no_television', "Number of VCD/DVD": 'no_cd_vcd_dvd',
            "Number of Component Stereo": 'no_component_stereo', "Number of Refrigerator": 'no_ref',
            "Number of Washing Machine": 'no_washingmachine', "Number of Air Conditioner": 'no_airconditioner',
            "Number of Car/Jeep/Van": 'no_car_jeep_van',
            "Number of Landline/Wireless Connection": 'no_landline_wireless',
            "Number of Phone": 'no_cp', "Number of Computer": 'no_pc',
            "Number of Gas Stove": 'no_stovegas', "Number of Bangka": 'no_banca',
            "Number of Motorcycle": 'no_motorcycle', "Poor": 'Poor'}

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True,
                cmap='coolwarm', xticklabels=d_hp, yticklabels=d_hp)

    plt.title('Correlation Between Family Possessions & Poor Status\n',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    heatmap_possessions_url = base64.b64encode(img.getvalue()).decode('utf8')

    # First Half of Income & Family
    # Related Expense Correlation
    df = pd.read_csv('datasets/fies-cleaned.csv')
    df = df.loc[df['region'].str.contains(region_value)]

    to_drop = ['region',
               'alcohol_expense',
               'tobacco_expense',
               'wear_expense',
               'houseing_water_expense',
               'imp_houserental',
               'medcare_expense',
               'transpo_expense',
               'commu_expense',
               'edu_expense',
               'misc_expense',
               'specialoc_expense',
               'farming_garden_expense',
               'totalincome_entrep',
               'head_sex',
               'head_age',
               'head_marital',
               'head_highestgrade',
               'head_job_business',
               'head_occupation',
               'head_class_worker',
               'householdtype',
               'total_fam_mem',
               'memage_less5',
               'memage_5-17',
               'no_employedfam',
               'bldghousetype',
               'rooftype',
               'walltype',
               'floor_area',
               'house_age',
               'no_bedrooms',
               'tenure',
               'toilet_facility',
               'electricity',
               'water_supply',
               'no_television',
               'no_cd_vcd_dvd',
               'no_component_stereo',
               'no_ref',
               'no_washingmachine',
               'no_airconditioner',
               'no_car_jeep_van',
               'no_landline_wireless',
               'no_cp',
               'no_pc',
               'no_stovegas',
               'no_banca',
               'no_motorcycle']
    df.drop(to_drop, inplace=True, axis=1)

    mainsrcTypeDummies = pd.get_dummies(df['mainsrc'])
    classTypeDummies = pd.get_dummies(df['class'])

    df = pd.concat([df, mainsrcTypeDummies], axis='columns')
    df = pd.concat([df, classTypeDummies], axis='columns')

    to_drop = ['class',
               'mainsrc',
               'Not Poor']
    df.drop(to_drop, inplace=True, axis=1)

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), annot=True,
                cmap='coolwarm')

    plt.title('Correlation Between (Income & Family Related Expense) & Poor Status\n',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    heatmap_incomeFamilyExpense1_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Second Half of Income & Family
    # Related Expense Correlation
    df = pd.read_csv('datasets/fies-cleaned.csv')
    df = df.loc[df['region'].str.contains(region_value)]

    to_drop = ['household_income',
               'region',
               'total_expenditure',
               'food_expense',
               'mainsrc',
               'agri_indicator',
               'bread_cereals_expense',
               'rice_expense',
               'meat_expense',
               'fish_marine_expense',
               'fruit_expense',
               'veg_expense',
               'restohotel_expense',
               'head_sex',
               'head_age',
               'head_marital',
               'head_highestgrade',
               'head_job_business',
               'head_occupation',
               'head_class_worker',
               'householdtype',
               'total_fam_mem',
               'memage_less5',
               'memage_5-17',
               'no_employedfam',
               'bldghousetype',
               'rooftype',
               'walltype',
               'floor_area',
               'house_age',
               'no_bedrooms',
               'tenure',
               'toilet_facility',
               'electricity',
               'water_supply',
               'no_television',
               'no_cd_vcd_dvd',
               'no_component_stereo',
               'no_ref',
               'no_washingmachine',
               'no_airconditioner',
               'no_car_jeep_van',
               'no_landline_wireless',
               'no_cp',
               'no_pc',
               'no_stovegas',
               'no_banca',
               'no_motorcycle']
    df.drop(to_drop, inplace=True, axis=1)

    classTypeDummies = pd.get_dummies(df['class'])

    df = pd.concat([df, classTypeDummies], axis='columns')

    to_drop = ['class',
               'Not Poor']
    df.drop(to_drop, inplace=True, axis=1)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True,
                cmap='coolwarm')

    plt.title('Correlation Between (Income & Family Related Expense) & Poor Status\n',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    heatmap_incomeFamilyExpense2_url = base64.b64encode(img.getvalue()).decode('utf8')

    # First Half of Income & Family
    # Related Expense Correlation
    df = pd.read_csv('datasets/fies-cleaned.csv')
    df = df.loc[df['region'].str.contains(region_value)]

    to_drop = ['household_income',
               'region',
               'total_expenditure',
               'food_expense',
               'mainsrc',
               'agri_indicator',
               'bread_cereals_expense',
               'rice_expense',
               'meat_expense',
               'fish_marine_expense',
               'fruit_expense',
               'veg_expense',
               'restohotel_expense',
               'alcohol_expense',
               'tobacco_expense',
               'wear_expense',
               'houseing_water_expense',
               'imp_houserental',
               'medcare_expense',
               'transpo_expense',
               'commu_expense',
               'edu_expense',
               'misc_expense',
               'specialoc_expense',
               'farming_garden_expense',
               'totalincome_entrep',
               'bldghousetype',
               'rooftype',
               'walltype',
               'floor_area',
               'house_age',
               'no_bedrooms',
               'tenure',
               'toilet_facility',
               'electricity',
               'water_supply',
               'no_television',
               'no_cd_vcd_dvd',
               'no_component_stereo',
               'no_ref',
               'no_washingmachine',
               'no_airconditioner',
               'no_car_jeep_van',
               'no_landline_wireless',
               'no_cp',
               'no_pc',
               'no_stovegas',
               'no_banca',
               'no_motorcycle']
    df.drop(to_drop, inplace=True, axis=1)

    headSexDummies = pd.get_dummies(df['head_sex'])
    headMaritalDummies = pd.get_dummies(df['head_marital'])
    headJobBusinessDummies = pd.get_dummies(df['head_job_business'])
    classTypeDummies = pd.get_dummies(df['class'])

    df = pd.concat([df, headSexDummies], axis='columns')
    df = pd.concat([df, headMaritalDummies], axis='columns')
    df = pd.concat([df, headJobBusinessDummies], axis='columns')
    df = pd.concat([df, classTypeDummies], axis='columns')

    to_drop = ['head_sex',
               'head_marital',
               'head_job_business',
               'householdtype',
               'class',
               'Not Poor']
    df.drop(to_drop, inplace=True, axis=1)

    df['head_highestgrade'] = df["head_highestgrade"].astype('category')
    df['head_occupation'] = df["head_occupation"].astype('category')
    df['head_class_worker'] = df["head_class_worker"].astype('category')

    df['head_highestgrade_cat'] = df["head_highestgrade"].cat.codes
    df['head_occupation_cat'] = df["head_occupation"].cat.codes
    df['head_classworker_cat'] = df["head_class_worker"].cat.codes

    to_drop = ['head_highestgrade',
               'head_occupation',
               'head_class_worker']
    df.drop(to_drop, inplace=True, axis=1)

    to_drop = ['No Job/Business',
               'With Job/Business',
               'head_highestgrade_cat',
               'head_occupation_cat',
               'head_classworker_cat']
    df.drop(to_drop, inplace=True, axis=1)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True,
                cmap='coolwarm')

    plt.title('Correlation Between Household Head & Poor Status\n',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    heatmap_householdHead1_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Second Half of Income & Family
    # Related Expense Correlation
    df = pd.read_csv('datasets/fies-cleaned.csv')
    df = df.loc[df['region'].str.contains(region_value)]

    to_drop = ['household_income',
               'region',
               'total_expenditure',
               'food_expense',
               'mainsrc',
               'agri_indicator',
               'bread_cereals_expense',
               'rice_expense',
               'meat_expense',
               'fish_marine_expense',
               'fruit_expense',
               'veg_expense',
               'restohotel_expense',
               'alcohol_expense',
               'tobacco_expense',
               'wear_expense',
               'houseing_water_expense',
               'imp_houserental',
               'medcare_expense',
               'transpo_expense',
               'commu_expense',
               'edu_expense',
               'misc_expense',
               'specialoc_expense',
               'farming_garden_expense',
               'totalincome_entrep',
               'bldghousetype',
               'rooftype',
               'walltype',
               'floor_area',
               'house_age',
               'no_bedrooms',
               'tenure',
               'toilet_facility',
               'electricity',
               'water_supply',
               'no_television',
               'no_cd_vcd_dvd',
               'no_component_stereo',
               'no_ref',
               'no_washingmachine',
               'no_airconditioner',
               'no_car_jeep_van',
               'no_landline_wireless',
               'no_cp',
               'no_pc',
               'no_stovegas',
               'no_banca',
               'no_motorcycle']
    df.drop(to_drop, inplace=True, axis=1)

    headSexDummies = pd.get_dummies(df['head_sex'])
    headMaritalDummies = pd.get_dummies(df['head_marital'])
    headJobBusinessDummies = pd.get_dummies(df['head_job_business'])
    householdtypeDummies = pd.get_dummies(df['householdtype'])
    classTypeDummies = pd.get_dummies(df['class'])

    df = pd.concat([df, headSexDummies], axis='columns')
    df = pd.concat([df, headMaritalDummies], axis='columns')
    df = pd.concat([df, headJobBusinessDummies], axis='columns')
    df = pd.concat([df, householdtypeDummies], axis='columns')
    df = pd.concat([df, classTypeDummies], axis='columns')

    to_drop = ['head_sex',
               'head_marital',
               'head_job_business',
               'householdtype',
               'class',
               'Not Poor']
    df.drop(to_drop, inplace=True, axis=1)

    df['head_highestgrade'] = df["head_highestgrade"].astype('category')
    df['head_occupation'] = df["head_occupation"].astype('category')
    df['head_class_worker'] = df["head_class_worker"].astype('category')

    df['head_highestgrade_cat'] = df["head_highestgrade"].cat.codes
    df['head_occupation_cat'] = df["head_occupation"].cat.codes
    df['head_classworker_cat'] = df["head_class_worker"].cat.codes

    to_drop = ['head_highestgrade',
               'head_occupation',
               'head_class_worker']
    df.drop(to_drop, inplace=True, axis=1)

    to_drop = ['head_age',
               'total_fam_mem',
               'memage_less5',
               'memage_5-17',
               'no_employedfam',
               'Female',
               'Male',
               'Annulled',
               'Divorced/Separated',
               'Married',
               'Single',
               'Widowed']
    df.drop(to_drop, inplace=True, axis=1)

    plt.figure(figsize=(10, 7))
    sns.heatmap(df.corr(), annot=True,
                cmap='coolwarm')

    plt.title('Correlation Between Household Head & Poor Status\n',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    heatmap_householdHead2_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('plot.html', bar_poor_url=bar_poor_url, pi_mainsrc_url=pi_mainsrc_url,
                           pi_status_url=pi_status_url, region_value=region_value,
                           pi_toilet_fac_url=pi_toilet_fac_url, pi_tenure_url=pi_tenure_url,
                           pi_wtr_sply_url=pi_wtr_sply_url,
                           poor_household_expenditures=poor_household_expenditures,
                           pr_expenditure_rgn=pr_expenditure_rgn,
                           not_poor_household_expenditures=not_poor_household_expenditures,
                           bar_typ_house_url=bar_typ_house_url,
                           bar_head_attainment_url=bar_head_attainment_url,
                           household_possessions_url=household_possessions_url,
                           heatmap_possessions_url=heatmap_possessions_url,
                           heatmap_incomeFamilyExpense1_url=heatmap_incomeFamilyExpense1_url,
                           heatmap_incomeFamilyExpense2_url=heatmap_incomeFamilyExpense2_url,
                           heatmap_householdHead1_url=heatmap_householdHead1_url,
                           heatmap_householdHead2_url=heatmap_householdHead2_url)


if __name__ == '__main__':
    app.run()
