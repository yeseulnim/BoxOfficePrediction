import urllib.request
import json
from datetime import date
from datetime import timedelta
from datetime import datetime
from time import sleep
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def get_weekend_box_office(end_date:date,numdays:int, noncommercial = "Y", nation = "F", filename = f"{datetime.now()}.json"):
    start = datetime.now()
    date_list = [str(end_date - timedelta(days=x)).replace('-', '') for x in range(0, numdays, 7)]
    # print(f"dates:{date_list[-1]}~{date_list[0]}")
    data = []

    for date in date_list:
        path = ("http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchWeeklyBoxOfficeList.json"
                + "?key=d3e95adbe4f2171c5c869d03afa93dae" # keys : d3e95adbe4f2171c5c869d03afa93dae / 9511b15ed35f976dff1642647019125c
                + "&multiMovieYn=" + noncommercial
                + "&repNationCd=" + nation
                + "&targetDt=" + date)

        for attempt in range(5):
            try:
                with urllib.request.urlopen(path) as url:
                    original_data = json.load(url)
                    for i in range(10):
                        try: # try-except because some weeks have less than 10 rows
                            data.append([date,
                                     original_data["boxOfficeResult"]['yearWeekTime'],
                                     i,
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["movieCd"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["movieNm"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["openDt"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["salesAmt"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["salesShare"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["salesInten"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["salesChange"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["salesAcc"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["audiCnt"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["audiInten"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["audiChange"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["audiAcc"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["scrnCnt"],
                                     original_data["boxOfficeResult"]["weeklyBoxOfficeList"][i]["showCnt"]
                                    ])
                        except:
                            pass
                break
            except HTTPError as e:
                if e.code == 503 and attempt < max_retries -1:
                    print(f"Attempt {attempt+1} failed. Retrying in 5 seconds...")
                    sleep(5)
    with open('data/'+ filename,'w')as f:
        json.dump(data,f)
    f.close()
    print(f"runtime:{datetime.now() - start}")


def prepare_data(data):
    data = pd.DataFrame(data)
    data = data.rename(columns={0: "Date", 1: "Week", 2: "Rank", 3: "MovieCode", 4: "MovieName", 5: "OpenDate",
                                6: "SalesAmount", 7: "SalesShare", 8: "SalesInten", 9: "SalesChange", 10: "SalesAcc",
                                11: "AudienceCount", 12: "AudienceInten", 13: "AudienceChange", 14: "AudienceAcc",
                                15: "ScreenCount", 16: "ShowCount"})

    print(f"Raw data length:{len(data)}")
    # change data formats
    data["Date"] = pd.to_datetime(data["Date"]).dt.date  # change dates to datetime format
    data[["Rank", "SalesAmount", "SalesShare", "SalesInten",  # change numbers to int
          "SalesChange", "SalesAcc", "AudienceCount", "AudienceInten",
          "AudienceChange", "AudienceAcc", "ScreenCount", "ShowCount"]] = data[["Rank", "SalesAmount", "SalesShare",
                                                                                "SalesInten", "SalesChange", "SalesAcc",
                                                                                "AudienceCount", "AudienceInten",
                                                                                "AudienceChange", "AudienceAcc",
                                                                                "ScreenCount", "ShowCount"]].apply(
        pd.to_numeric)
    # some opendate values are empty strings
    # drop rows with empty opendate before changing format
    data.drop(data[(data["OpenDate"] == " ")].index, inplace=True)
    data["OpenDate"] = pd.to_datetime(data["OpenDate"]).dt.date

    # 2010 - 2019
    # drop rows where opening date is before 2010 or after 2019
    # data.drop(data[(data["OpenDate"] > date(2019, 12, 31))].index, inplace=True)
    # data.drop(data[(data["OpenDate"]) < date(2010, 1, 1)].index, inplace=True)

    # 2023
    # drop rows where opening date is before 2023 or after 2023
    # data.drop(data[(data["OpenDate"] > date(2023, 12, 31))].index, inplace=True)
    # data.drop(data[(data["OpenDate"]) < date(2023, 1, 1)].index, inplace=True)

    # 2020 Jan
    data.drop(data[(data["OpenDate"] > date(2020, 1, 31))].index, inplace=True)
    data.drop(data[(data["OpenDate"]) < date(2020, 1, 1)].index, inplace=True)

    # 2020 Feb
    # data.drop(data[(data["OpenDate"] > date(2020, 2, 29))].index, inplace=True)
    # data.drop(data[(data["OpenDate"]) < date(2020, 2, 1)].index, inplace=True)

    # 2020 Mar
    # data.drop(data[(data["OpenDate"] > date(2020, 3, 31))].index, inplace=True)
    # data.drop(data[(data["OpenDate"]) < date(2020, 3, 1)].index, inplace=True)


    print(f"Cleaned data length:{len(data)}")
    return data



def get_movie_info(moviecode_list, filename = f"{datetime.now()}.json"):
    start = datetime.now()
    data = []

    for moviecode in moviecode_list:
        moviecode = str(moviecode)
        path = ("http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json"
                + "?key=9511b15ed35f976dff1642647019125c" # keys : d3e95adbe4f2171c5c869d03afa93dae / 9511b15ed35f976dff1642647019125c
                + "&movieCd=" + moviecode)
        with urllib.request.urlopen(path) as url:
            original_data = json.load(url)
            data.append([moviecode,
                             original_data["movieInfoResult"]["movieInfo"]['movieNm'],
                             original_data["movieInfoResult"]["movieInfo"]['movieNmEn'],
                             original_data["movieInfoResult"]["movieInfo"]['showTm'],
                             original_data["movieInfoResult"]["movieInfo"]['prdtYear'],
                             original_data["movieInfoResult"]["movieInfo"]['openDt'],
                             original_data["movieInfoResult"]["movieInfo"]['prdtStatNm'],
                             original_data["movieInfoResult"]["movieInfo"]['typeNm'],
                             original_data["movieInfoResult"]["movieInfo"]["nations"][0]["nationNm"],
                             original_data["movieInfoResult"]["movieInfo"]["genres"],
                             original_data["movieInfoResult"]["movieInfo"]["companys"],
                             original_data["movieInfoResult"]["movieInfo"]["audits"][0]["watchGradeNm"],
                             original_data["movieInfoResult"]["movieInfo"]["staffs"]
                            ])
    with open('data/'+ filename,'w')as f:
        json.dump(data,f)
    f.close()
    print(f"runtime:{datetime.now() - start}")


def extract_movie_info(movie_data):
    genres = tuple(genre['genreNm'] for genre in movie_data[9])

    production_companies = set(
        company['companyNm']
        for company in movie_data[10]
        if company['companyPartNm'] == '제작사'
    )

    distribution_companies = set(
        company['companyNm']
        for company in movie_data[10]
        if company['companyPartNm'] == '배급사'
    )

    importation_companies = set(
        company['companyNm']
        for company in movie_data[10]
        if company['companyPartNm'] == '수입사'
    )

    investor_count = len(set(
        company['companyNm']
        for company in movie_data[10]
        if company['companyPartNm'] in ('제공','공동제공')
    ))

    company_count = len(set(
        company['companyNm']
        for company in movie_data[10]
    ))

    staff_count = len(movie_data[12])

    return{
        'movie_code' : movie_data[0],
        'movie_name' : movie_data[1],
        'movie_name_en' : movie_data[2],
        'runtime' : movie_data[3],
        'prod_year' : movie_data[4],
        'open_date' : movie_data[5],
        'prod_stat' : movie_data[6],
        'type' : movie_data[7],
        'nation' : movie_data[8],
        'genre' : genres,
        'production_companies' : production_companies,
        'distribution_companies' : distribution_companies,
        'importation_companies' : importation_companies,
        'investor_count' : investor_count,
        'company_count' : company_count,
        'staff_count' : staff_count,
        'rating' : movie_data[11]
    }

def categorize_companies(companies):
    categorized = []
    for company in companies:
        # standardize names to major distributors & etc
        company = standardize_company_name(company)
        categorized.append(company)
    return tuple(set(categorized))

def standardize_company_name(company):
    if company in ['롯데쇼핑㈜롯데시네마', '롯데쇼핑㈜롯데엔터테인먼트', '롯데컬처웍스(주)롯데엔터테인먼트']:
        return 'Lotte'
    elif company in ['(주)씨제이이엔엠', 'CJ ENM']:
        return 'CJ ENM'
    elif company in ['씨너스엔터테인먼트(주)', '플러스엠 엔터테인먼트']:
        return 'PlusM'
    elif company == '(주)쇼박스':
        return 'ShowBox'
    elif company == '(주)넥스트엔터테인먼트월드(NEW)':
        return 'NEW'
    elif company in ['워너 브러더스 픽쳐스', '워너브러더스 코리아(주)', '워너브러더스사㈜']:
        return 'Warner'
    elif company in ['월트디즈니컴퍼니코리아 유한책임회사', '월트디즈니컴퍼니코리아(주)', '소니픽쳐스릴리징월트디즈니스튜디오스코리아(주)']:
        return 'Disney'
    elif company == '유니버설픽쳐스인터내셔널 코리아(유)':
        return 'Universal'
    else:
        return 'etc'

def standardize_nation_name(nation):
    if nation == '한국':
        return 'Korea'
    elif nation == '미국':
        return 'USA'
    elif nation == '일본':
        return 'Japan'
    else:
        return 'etc'

def standardize_type_name(type):
    if type == '장편':
        return 'longform'
    else:
        return 'etc'

def standardize_ratings(rating):
    if rating == '전체관람가':
        return 'all'
    elif rating == '12세이상관람가':
        return '12'
    elif rating == '15세이상관람가':
        return '15'
    elif rating == '청소년관람불가':
        return '18'
    else:
        return 'etc'


def one_hot_encode_column(df, column_name, prefix):
    # Convert the column to a list of lists
    series = df[column_name].fillna('').apply(lambda x: [x] if isinstance(x, str) else x)

    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(series)
    encoded_df = pd.DataFrame(encoded, columns=[f"{prefix}_{cls}" for cls in mlb.classes_], index=df.index)

    # Drop the original column and concatenate the encoded columns
    result_df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)

    return result_df