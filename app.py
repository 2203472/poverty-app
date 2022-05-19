import math

import flask
import pickle
import pandas as pd
from scipy.stats import stats
from sklearn import preprocessing
import seaborn as sns

from flask import Flask, render_template, send_file, request, url_for, redirect
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template
from io import BytesIO
import base64

# Use pickle to load in the pre-trained model.
with open(f'model/poverty-prediction-rforest3.pkl', 'rb') as f:
    model = pickle.load(f)

app = flask.Flask(__name__,
                  static_folder='static',
                  template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    with open(f'datasets/raw-features.pkl', 'rb') as f:
        df = pickle.load(f)

    if flask.request.method == 'GET':
        return flask.render_template('main.html')

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
            # alcohol_expense = flask.request.form['alcoholexp']
            # tobacco_expense = flask.request.form['tobaccoexp']
            wear_expense = float(flask.request.form['wearexp'])
            housing_water_expense = float(flask.request.form['housingwaterexp'])
            imp_houserental = float(flask.request.form['imphouserental'])
            transpo_expense = float(flask.request.form['transpoexp'])
            commu_expense = float(flask.request.form['commuexp'])
            edu_expense = float(flask.request.form['eduexp'])
            misc_expense = float(flask.request.form['miscexp'])
            # specialoc_expense = flask.request.form['specoccexp']
            farming_garden_expense = float(flask.request.form['farmgardenexp'])

            food_expense = float(bread_cereals_expense) + float(rice_expense) + float(meat_expense) + float(
                fish_marine_expense) + float(fruit_expense) + float(veg_expense)

            df.loc[len(df)] = [household_income, food_expense, bread_cereals_expense, meat_expense, rice_expense,
                               fruit_expense,
                               fish_marine_expense, veg_expense, restohotel_expense, wear_expense,
                               housing_water_expense, imp_houserental, transpo_expense,
                               commu_expense, edu_expense, misc_expense, farming_garden_expense, 5]

            # Normalization
            x = df.values
            min_max_scaler = preprocessing.MinMaxScaler()
            df = pd.DataFrame(min_max_scaler.fit_transform(x), index=df.index, columns=df.columns)
            input_variables = df.tail(1)

            prediction = model.predict(input_variables)[0]

            if prediction == 0:
                printpredict = "Not Poor"
            else:
                printpredict = "Poor"

            return flask.render_template('main.html', result=printpredict)

        if request.form['btn'] == 'Plot':
            region = request.form["region"]
            return redirect(url_for("plot", rgn=region))

    return render_template("main.html")


@app.route('/<rgn>')
def plot(rgn):
    df = pd.read_csv('datasets/fies-cleaned.csv')
    region_value = rgn
    img = BytesIO()

    # Poor Average Household income
    poor_data = df.loc[df['class'].str.contains("Poor") & df['region'].str.contains(region_value)]
    rgn_poor_household_income = poor_data['household_income']
    poor_household_income = "{:.2f}".format(rgn_poor_household_income.mean())

    # Not Poor Average Household income
    not_poor_data = df.loc[df['class'].str.contains('Not Poor') & df['region'].str.contains(region_value)]
    rgn_not_poor_household_income = not_poor_data['household_income']
    not_poor_household_income = "{:.2f}".format(rgn_not_poor_household_income.mean())

    #  Poor Occupation count per region
    poor_occupation_count = poor_data['head_occupation'].value_counts()
    poor_occupation_count = poor_occupation_count[:10, ]

    plt.figure(figsize=(15, 5))
    sns.barplot(poor_occupation_count.index, poor_occupation_count.values, alpha=0.5,
                hue=poor_occupation_count.index, dodge=False)
    plt.title('Top 10 Occupations of Poor People')
    plt.ylabel('Occurrence')
    plt.xticks([])

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    bar_poor_url = base64.b64encode(img.getvalue()).decode('utf8')

    #  Not poor Occupation count per region
    not_poor_occupation_count = not_poor_data['head_occupation'].value_counts()
    not_poor_occupation_count = not_poor_occupation_count[:10, ]

    plt.figure(figsize=(15, 5))
    sns.barplot(not_poor_occupation_count.index, not_poor_occupation_count.values, alpha=0.5,
                hue=not_poor_occupation_count.index, dodge=False)
    plt.title('Top 10 Occupations of Not Poor People')
    plt.ylabel('Occurrence')
    plt.xticks([])

    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    bar_not_poor_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Gender count of each region
    region_data = df.loc[df['region'].str.contains(region_value)]

    gender_count = region_data['head_sex'].value_counts()
    explode = (0.1, 0.1)
    key_list = list(gender_count.keys())

    values = []
    for key in gender_count.keys():
        values.append(gender_count[key])

    colors = ['#df9d9e', '#c8b8d8', '#c1aca8']

    plt.pie(values, labels=key_list, shadow=True, startangle=180, explode=explode, autopct='%1.1f%%', colors=colors)

    plt.title('GENDER',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_gender_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Financial Status Section
    status_count = region_data['class'].value_counts()
    status_list = list(status_count.keys())

    values_status = []
    for key in status_count.keys():
        values_status.append(status_count[key])

    plt.pie(values_status, labels=status_list,
            shadow=True, startangle=180,
            explode=explode, autopct='%1.1f%%',
            colors=colors)

    plt.title('FINANCIAL STATUS',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_status_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Household Type
    household_type_count = region_data['householdtype'].value_counts()
    household_type_list = list(household_type_count.keys())

    household_type_values = []
    for key in household_type_count.keys():
        household_type_values.append(household_type_count[key])

    plt.pie(household_type_values, labels=household_type_list,
            shadow=True, startangle=180,
            autopct='%1.1f%%', colors=colors)

    plt.title('HOUSEHOLD TYPE',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_household_type_url = base64.b64encode(img.getvalue()).decode('utf8')

    # Top 3 Valued Numerical Features Based on Pearson Correlation Section
    regionData = df.loc[df['region'].str.contains(region_value)]

    to_drop = ['region',
               'mainsrc',
               'head_sex',
               'head_marital',
               'head_highestgrade',
               'head_job_business',
               'head_occupation',
               'head_class_worker',
               'householdtype',
               'bldghousetype',
               'rooftype',
               'walltype',
               'tenure',
               'toilet_facility',
               'water_supply']

    region_data.drop(to_drop, inplace=True, axis=1)

    classTypeDummies = pd.get_dummies(regionData['class'])

    regionData = pd.concat([regionData, classTypeDummies], axis='columns')

    to_drop = ['class',
               'Poor']
    regionData.drop(to_drop, inplace=True, axis=1)

    pearsonsRCoefficientTelevision, pValue = stats.pearsonr(
        regionData['no_television'], regionData['Not Poor'])

    pearsonsRCoefficientMultimedia, pValue = stats.pearsonr(
        regionData['no_cd_vcd_dvd'], regionData['Not Poor'])

    pearsonsRCoefficientStereo, pValue = stats.pearsonr(
        regionData['no_component_stereo'], regionData['Not Poor'])

    pearsonsRCoefficientRefrigerator, pValue = stats.pearsonr(
        regionData['no_ref'], regionData['Not Poor'])

    pearsonsRCoefficientWashingMachine, pValue = stats.pearsonr(
        regionData['no_washingmachine'], regionData['Not Poor'])

    pearsonsRCoefficientAircon, pValue = stats.pearsonr(
        regionData['no_airconditioner'], regionData['Not Poor'])

    pearsonsRCoefficientAutomobile, pValue = stats.pearsonr(
        regionData['no_car_jeep_van'], regionData['Not Poor'])

    pearsonsRCoefficientWirelessLandline, pValue = stats.pearsonr(
        regionData['no_landline_wireless'], regionData['Not Poor'])

    pearsonsRCoefficientCellphone, pValue = stats.pearsonr(
        regionData['no_cp'], regionData['Not Poor'])

    pearsonsRCoefficientComputer, pValue = stats.pearsonr(
        regionData['no_pc'], regionData['Not Poor'])

    pearsonsRCoefficientGasStove, pValue = stats.pearsonr(
        regionData['no_stovegas'], regionData['Not Poor'])

    pearsonsRCoefficientBangka, pValue = stats.pearsonr(
        regionData['no_banca'], regionData['Not Poor'])

    if math.isnan(pearsonsRCoefficientBangka) is True:
        pearsonsRCoefficientBangka = 0

    pearsonsRCoefficientMotorcycle, pValue = stats.pearsonr(
        regionData['no_motorcycle'], regionData['Not Poor'])

    pearsonsRCoefficientBreadCereal, pValue = stats.pearsonr(
        regionData['bread_cereals_expense'], regionData['Not Poor'])

    pearsonsRCoefficientRice, pValue = stats.pearsonr(
        regionData['rice_expense'], regionData['Not Poor'])

    pearsonsRCoefficientMeat, pValue = stats.pearsonr(
        regionData['meat_expense'], regionData['Not Poor'])

    pearsonsRCoefficientFish, pValue = stats.pearsonr(
        regionData['fish_marine_expense'], regionData['Not Poor'])

    pearsonsRCoefficientFruit, pValue = stats.pearsonr(
        regionData['fruit_expense'], regionData['Not Poor'])

    pearsonsRCoefficientVegetable, pValue = stats.pearsonr(
        regionData['veg_expense'], regionData['Not Poor'])

    pearsonsRCoefficientHouseholdIncome, pValue = stats.pearsonr(
        regionData['household_income'], regionData['Not Poor'])

    pearsonsRCoefficientFood, pValue = stats.pearsonr(
        regionData['food_expense'], regionData['Not Poor'])

    pearsonsRCoefficientAgriculture, pValue = stats.pearsonr(
        regionData['agri_indicator'], regionData['Not Poor'])

    pearsonsRCoefficientRestohotel, pValue = stats.pearsonr(
        regionData['restohotel_expense'], regionData['Not Poor'])

    pearsonsRCoefficientAlcohol, pValue = stats.pearsonr(
        regionData['alcohol_expense'], regionData['Not Poor'])

    pearsonsRCoefficientTobacco, pValue = stats.pearsonr(
        regionData['tobacco_expense'], regionData['Not Poor'])

    pearsonsRCoefficientWear, pValue = stats.pearsonr(
        regionData['wear_expense'], regionData['Not Poor'])

    pearsonsRCoefficientHouseingWater, pValue = stats.pearsonr(
        regionData['houseing_water_expense'], regionData['Not Poor'])

    pearsonsRCoefficientHouseRental, pValue = stats.pearsonr(
        regionData['imp_houserental'], regionData['Not Poor'])

    pearsonsRCoefficientMedcare, pValue = stats.pearsonr(
        regionData['medcare_expense'], regionData['Not Poor'])

    pearsonsRCoefficientTranspo, pValue = stats.pearsonr(
        regionData['transpo_expense'], regionData['Not Poor'])

    pearsonsRCoefficientCommu, pValue = stats.pearsonr(
        regionData['commu_expense'], regionData['Not Poor'])

    pearsonsRCoefficientEducational, pValue = stats.pearsonr(
        regionData['edu_expense'], regionData['Not Poor'])

    pearsonsRCoefficientMiscelanous, pValue = stats.pearsonr(
        regionData['misc_expense'], regionData['Not Poor'])

    pearsonsRCoefficientSpecialLoc, pValue = stats.pearsonr(
        regionData['specialoc_expense'], regionData['Not Poor'])

    pearsonsRCoefficientFarmingGarden, pValue = stats.pearsonr(
        regionData['farming_garden_expense'], regionData['Not Poor'])

    pearsonsRCoefficientTotalIncomeEntrep, pValue = stats.pearsonr(
        regionData['totalincome_entrep'], regionData['Not Poor'])

    pearsonsRCoefficientHeadAge, pValue = stats.pearsonr(
        regionData['head_age'], regionData['Not Poor'])

    pearsonsRCoefficientTotalFamMem, pValue = stats.pearsonr(
        regionData['total_fam_mem'], regionData['Not Poor'])

    pearsonsRCoefficientMemage_less, pValue = stats.pearsonr(
        regionData['memage_less5'], regionData['Not Poor'])

    pearsonsRCoefficientMemage_5, pValue = stats.pearsonr(
        regionData['memage_5-17'], regionData['Not Poor'])

    pearsonsRCoefficientEmployedFam, pValue = stats.pearsonr(
        regionData['no_employedfam'], regionData['Not Poor'])

    pearsonsRCoefficientFloorArea, pValue = stats.pearsonr(
        regionData['floor_area'], regionData['Not Poor'])

    pearsonsRCoefficientHouseAge, pValue = stats.pearsonr(
        regionData['house_age'], regionData['Not Poor'])

    pearsonsRCoefficientBedrooms, pValue = stats.pearsonr(
        regionData['no_bedrooms'], regionData['Not Poor'])

    pearsonsRCoefficientElectricity, pValue = stats.pearsonr(
        regionData['electricity'], regionData['Not Poor'])

    televisionScore = {"Number of Television": pearsonsRCoefficientTelevision}
    multimediaScore = {"Number of CD,VCD, and DVD": pearsonsRCoefficientMultimedia}
    stereoScore = {"Number of Stereo": pearsonsRCoefficientStereo}
    refrigeratorScore = {"Number of Refrigerator": pearsonsRCoefficientRefrigerator}
    washingMachingScore = {"Number of Washing Machine": pearsonsRCoefficientWashingMachine}
    airconScore = {"Number of Airconditioner": pearsonsRCoefficientAircon}
    automobileScore = {"Number of Automobile": pearsonsRCoefficientAutomobile}
    wirelessLandlineScore = {"Number of Wireless Landline": pearsonsRCoefficientWirelessLandline}
    cellphoneScore = {"Number of Cellphone": pearsonsRCoefficientCellphone}
    computerScore = {"Number of Computer": pearsonsRCoefficientComputer}
    gasStoveScore = {"Number of Gas Stove": pearsonsRCoefficientGasStove}
    bangkaScore = {"Number of Bangka": pearsonsRCoefficientBangka}
    motorcycleScore = {"Number of Motorcycle": pearsonsRCoefficientMotorcycle}
    breadCerealScore = {"Bread and Cereals Expense": pearsonsRCoefficientBreadCereal}
    riceScore = {"Rice Expense": pearsonsRCoefficientRice}
    meatScore = {"Meat Expense": pearsonsRCoefficientMeat}
    fishScore = {"Fish Expense": pearsonsRCoefficientFish}
    fruitScore = {"Fruit Expense": pearsonsRCoefficientFruit}
    vegetableScore = {"Vegetable Expense": pearsonsRCoefficientVegetable}
    householdIncomeScore = {"Household Income": pearsonsRCoefficientHouseholdIncome}
    foodScore = {"Food Expense": pearsonsRCoefficientFood}
    agricultureScore = {"Agricultularal Indicator": pearsonsRCoefficientAgriculture}
    restoHotelScore = {"Restaurant-Hotel Expense": pearsonsRCoefficientRestohotel}
    alcoholScore = {"Alcohol Expense": pearsonsRCoefficientAlcohol}
    tobaccoScore = {"Tobacco Expense": pearsonsRCoefficientTobacco}
    wearScore = {"Wear Expense": pearsonsRCoefficientWear}
    houseingWaterScore = {"Houseing Water Expense": pearsonsRCoefficientHouseingWater}
    houseRentalScore = {"House Rental Expense": pearsonsRCoefficientHouseRental}
    medcareScore = {"Medcare Expense": pearsonsRCoefficientMedcare}
    transpoScore = {"Transportation Expense": pearsonsRCoefficientTranspo}
    commuScore = {"Communication Expense": pearsonsRCoefficientCommu}
    educationalScore = {"Educational Expense": pearsonsRCoefficientEducational}
    miscelanousScore = {"Miscelanous Expense": pearsonsRCoefficientMiscelanous}
    specialLocScore = {"Special Loc Expense": pearsonsRCoefficientSpecialLoc}
    farmingGardenScore = {"Farming Garden Expense": pearsonsRCoefficientFarmingGarden}
    totalIncomeEntrepScore = {"Total Income Entrepreneur Expense": pearsonsRCoefficientTotalIncomeEntrep}
    headAgeScore = {"Head Age": pearsonsRCoefficientHeadAge}
    famMemScore = {"Total Family Members": pearsonsRCoefficientTotalFamMem}
    memage_lessScore = {"memage_less5": pearsonsRCoefficientMemage_less}
    memage_5Score = {"memage_5-17": pearsonsRCoefficientMemage_5}
    employedFamScore = {"Number of Employed Family Member": pearsonsRCoefficientEmployedFam}
    floorAreaScore = {"Floor Area": pearsonsRCoefficientFloorArea}
    houseAgeScore = {"House Age": pearsonsRCoefficientHouseAge}
    bedroomsScore = {"Number of Bedrooms": pearsonsRCoefficientBedrooms}
    electricityScore = {"Electricity": pearsonsRCoefficientElectricity}

    pearsonScore = dict()

    pearsonScore.update(televisionScore)
    pearsonScore.update(multimediaScore)
    pearsonScore.update(stereoScore)
    pearsonScore.update(refrigeratorScore)
    pearsonScore.update(washingMachingScore)
    pearsonScore.update(airconScore)
    pearsonScore.update(automobileScore)
    pearsonScore.update(wirelessLandlineScore)
    pearsonScore.update(cellphoneScore)
    pearsonScore.update(computerScore)
    pearsonScore.update(gasStoveScore)
    pearsonScore.update(bangkaScore)
    pearsonScore.update(motorcycleScore)
    pearsonScore.update(breadCerealScore)
    pearsonScore.update(riceScore)
    pearsonScore.update(meatScore)
    pearsonScore.update(fishScore)
    pearsonScore.update(fruitScore)
    pearsonScore.update(vegetableScore)
    pearsonScore.update(householdIncomeScore)
    pearsonScore.update(foodScore)
    pearsonScore.update(agricultureScore)
    pearsonScore.update(restoHotelScore)
    pearsonScore.update(alcoholScore)
    pearsonScore.update(tobaccoScore)
    pearsonScore.update(wearScore)
    pearsonScore.update(houseingWaterScore)
    pearsonScore.update(houseRentalScore)
    pearsonScore.update(medcareScore)
    pearsonScore.update(transpoScore)
    pearsonScore.update(commuScore)
    pearsonScore.update(educationalScore)
    pearsonScore.update(miscelanousScore)
    pearsonScore.update(specialLocScore)
    pearsonScore.update(farmingGardenScore)
    pearsonScore.update(totalIncomeEntrepScore)
    pearsonScore.update(headAgeScore)
    pearsonScore.update(famMemScore)
    pearsonScore.update(memage_lessScore)
    pearsonScore.update(memage_5Score)
    pearsonScore.update(employedFamScore)
    pearsonScore.update(floorAreaScore)
    pearsonScore.update(houseAgeScore)
    pearsonScore.update(bedroomsScore)
    pearsonScore.update(electricityScore)

    highestKeys = sorted(pearsonScore, key=pearsonScore.get, reverse=True)[:3]

    values = []
    for key in highestKeys:
        values.append(pearsonScore[key])

    explode = (0.3, 0.1, 0.0)

    plt.axis('equal')
    plt.pie(values, labels=highestKeys,
            shadow=True, startangle=90,
            explode=explode, autopct='%1.1f%%',
            colors=colors)

    plt.title('Top 3 Features Based on Pearson R',
              fontname="Century Gothic",
              size=18)

    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close()
    img.seek(0)
    pi_top_numerical_features = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('plot.html', bar_poor_url=bar_poor_url, bar_not_poor_url=bar_not_poor_url,
                           pi_gender_url=pi_gender_url, pi_status_url=pi_status_url, region_value=region_value,
                           poor_household_income=poor_household_income,
                           not_poor_household_income=not_poor_household_income,
                           pi_household_type_url=pi_household_type_url,
                           pi_top_numerical_features_url=pi_top_numerical_features)


if __name__ == '__main__':
    app.run()
