import urllib.request
import json
from datetime import date
from datetime import timedelta
from datetime import datetime
import pandas as pd


def get_weekend_box_office(end_date:date,numdays:int, commercial = "Y", nation = "F", filename = f"{datetime.now()}.json"):
    start = datetime.now()
    date_list = [str(end_date - timedelta(days=x)).replace('-', '') for x in range(0, numdays, 7)]
    # print(f"dates:{date_list[-1]}~{date_list[0]}")
    data = []

    for date in date_list:
        path = ("http://kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchWeeklyBoxOfficeList.json"
                + "?key=9511b15ed35f976dff1642647019125c" # keys : d3e95adbe4f2171c5c869d03afa93dae / 9511b15ed35f976dff1642647019125c
                + "&multiMovieYn=" + commercial
                + "&repNationCd=" + nation
                + "&targetDt=" + date)
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

    # drop rows where opening date is before 2010 or after 2019
    data.drop(data[(data["OpenDate"] > date(2019, 12, 31))].index, inplace=True)
    data.drop(data[(data["OpenDate"]) < date(2010, 1, 1)].index, inplace=True)

    print(f"Cleaned data length:{len(data)}")
    return data


def get_movie_info(moviecode_list, filename = f"{datetime.now()}.json"):
    start = datetime.now()
    data = []

    for moviecode in moviecode_list:
        path = ("http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieInfo.json"
                + "?key=d3e95adbe4f2171c5c869d03afa93dae" # keys : d3e95adbe4f2171c5c869d03afa93dae / 9511b15ed35f976dff1642647019125c
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
