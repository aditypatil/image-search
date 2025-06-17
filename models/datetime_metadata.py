import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from PIL.ExifTags import TAGS, GPSTAGS
from pillow_heif import register_heif_opener
from tqdm import tqdm
from datetime import datetime, timezone, timedelta
import pickle
import re
import requests
import json
import subprocess
import atexit
from zoneinfo import ZoneInfo
import time

register_heif_opener()

# Get system's local timezone
import tzlocal
local_tz = tzlocal.get_localzone()
# Make datetime.now() timezone-aware using local timezone
local_now = datetime.now(ZoneInfo(str(local_tz)))
# print(local_tz)
# print(local_now)


class DateTimeExtractor:
    def __init__(self, embedding_dir='embed_store'):
        self.embedding_dir = embedding_dir
        self.default_tz = timezone(timedelta(hours=5, minutes=30))

    def __format_iso__(self, dt):
        # Ensure timezone-aware
        if dt.tzinfo is None:
            raise ValueError("datetime must be timezone-aware")
        # Format datetime with milliseconds and timezone
        base = dt.strftime('%Y-%m-%dT%H:%M:%S')
        offset = dt.strftime('%z')  # like "-0700"
        offset_colon = offset[:-2] + ':' + offset[-2:]  # format as "-07:00"
        return base + offset_colon

    
    def __get_file_date_creation__(self, image_path):
        """Fallback to date file was created if issues with EXIF"""
        stats = os.stat(image_path)
        created = datetime.fromtimestamp(stats.st_ctime)
        created = created.replace(tzinfo=self.default_tz)
        offset = created.strftime('%z')
        base = datetime.strptime(created, "%Y-%m-%dT%H:%M:%S")
        offset = offset[:-2] + ':' + offset[-2:]
        return base + offset

    def __extract_datetime__(self, image_path):
        """Extract datetime from EXIF"""
        datetime_extract = None
        try:
            image = Image.open(image_path)
            image.verify()
            if image_path.lower().endswith(('.heic', '.heif')):
                # Extract datetime from HEIC/HEIF image
                try:
                    # Try to get EXIF data from HEIC/HEIF
                    exif_dict = image.getexif()
                    if hasattr(exif_dict, 'get_ifd'):
                        exif_ifd = exif_dict.get_ifd(0x8769)  # EXIF IFD
                        dtexif = exif_ifd.get(36867)  # DateTimeOriginal
                        dtexif_zone = exif_ifd.get(36881)  # OffsetTimeOriginal
                    else:
                        dtexif = exif_dict.get(36867)
                        dtexif_zone = exif_dict.get(36881)
                except (AttributeError, KeyError):
                    dtexif = None
                    dtexif_zone = None
            else:
                # Extract datetime from regular image formats
                try:
                    exif_dict = image._getexif()
                    if exif_dict:
                        dtexif = exif_dict.get(36867)  # DateTimeOriginal
                        dtexif_zone = exif_dict.get(36881)  # OffsetTimeOriginal
                    else:
                        dtexif = None
                        dtexif_zone = None
                except AttributeError:
                    # Fallback to getexif() method
                    exif_dict = image.getexif()
                    dtexif = exif_dict.get(36867)
                    dtexif_zone = exif_dict.get(36881)
            
            if dtexif:
                # Format: "YYYY:MM:DD HH:MM:SS"
                dt = datetime.strptime(dtexif, '%Y-%m-%dT%H:%M:%S')
                
                if dtexif_zone:
                    # Parse timezone offset (format: "+05:30" or "-08:00")
                    try:
                        # Handle timezone offset string
                        if isinstance(dtexif_zone, str):
                            # Parse offset like "+05:30"
                            sign = 1 if dtexif_zone[0] == '+' else -1
                            hours, minutes = map(int, dtexif_zone[1:].split(':'))
                            offset = timezone(timedelta(hours=sign*hours, minutes=sign*minutes))
                            datetime_extract = dt.replace(tzinfo=offset)
                        else:
                            # If timezone format is unexpected, use IST as default
                            datetime_extract = dt.replace(tzinfo=self.default_tz)
                    except (ValueError, IndexError):
                        # If timezone parsing fails, use IST as default
                        datetime_extract = dt.replace(tzinfo=self.default_tz)
                else:
                    # If no timezone found in the image, use IST as default
                    datetime_extract = dt.replace(tzinfo=self.default_tz)

        except (UnidentifiedImageError, AttributeError, KeyError, OSError, ValueError) as e:
            datetime_extract = None
            # print(f"Error reading EXIF from {image_path}: \n{e}")
        
        if datetime_extract==None:
            return self.__format_iso__(self.__get_file_date_creation__(image_path))

        return self.__format_iso__(datetime_extract)

    def generate_datetime_metadata(self, image_paths):
        datetimes = []
        for img_path in tqdm(image_paths):
            dt = self.__extract_datetime__(img_path)
            datetimes.append(str(dt))
            # print(f"Image: {os.path.basename(img_path)}, Datetime:{dt}")
        with open(os.path.join(self.embedding_dir, 'datetime_metadata.npy'), 'wb') as f:
            np.save(f, datetimes)
        pass

class DucklingEngine:
    def __init__(self, port=8010, timezone="Asia/Kolkata", duckling_path="./duckling", timeout=60):
        self.port = port
        self.url = f"http://127.0.0.1:{self.port}/parse"  # Use 127.0.0.1 instead of 0.0.0.0
        self.timezone = timezone
        self.duckling_path = duckling_path
        self.duckling_process = None
        self.timeout = timeout
        
        # Start Duckling and wait for it to be ready
        self._start_duckling_server()
        atexit.register(self.cleanup_duckling)

    def _start_duckling_server(self):
        """Start Duckling server and wait for it to be ready"""
        print(f"Starting Duckling server on port {self.port}...")
        
        # Start the process
        self.duckling_process = subprocess.Popen(
            ["stack", "exec", "duckling-example-exe", "--", "--port", str(self.port)],
            cwd=self.duckling_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group for better cleanup
        )
        
        # Wait for server to be ready
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                # Try a simple health check
                response = requests.get(f"http://127.0.0.1:{self.port}", timeout=2)
                if response.status_code in [200, 404]:  # 404 is ok, means server is up
                    print(f"Duckling server ready in {time.time() - start_time:.1f} seconds")
                    return
            except requests.exceptions.RequestException:
                pass
            
            # Check if process died
            if self.duckling_process.poll() is not None:
                stdout, stderr = self.duckling_process.communicate()
                raise RuntimeError(f"Duckling process died: {stderr.decode()}")
            
            time.sleep(0.5)
        
        raise TimeoutError(f"Duckling server didn't start within {self.timeout} seconds")
    
    def __del__(self):
        self.cleanup_duckling()

    # Kill Duckling on exit
    def cleanup_duckling(self):
        if self.duckling_process.poll() is None:  # Still running
            self.duckling_process.terminate()
            print("Duckling server terminated.")
    
    def __replacer__(self, main_text, text, sub):
        return re.sub(re.escape(text), sub, main_text, flags=re.IGNORECASE)
    
    def __replace_methods__(self, query1):
        rep_list = [
            ["monsoon", "June to September"],
            ["independence day", "15th August"],
            ["republic day", "26th January"]
        ]
        for rep in rep_list:
            query1 = self.__replacer__(query1, rep[0], rep[1])
        query1 = query1.replace(",","")
        query1 = query1.replace("from ","")
        return query1
    
    def get_response(self, query):
        query = self.__replace_methods__(query)
        payload = {
            "locale": "en_IN",
            "text": query,
            "dims": ["time"],
            "tz": self.timezone,
        }
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        response = requests.post(self.url, data=payload, headers=headers)
        return response.json()

class DateSearch:
    def __init__(self, datebase):
        # self.find_holidays = find_holidays
        self.datebase = datebase
        self.grain_order = ["year", "month", "day", "hour", "minute", "second"]
        self.grainrep_year = 2000
        self.grainrep_month = 1
        self.grainrep_day = 1
        self.grainrep_hour = 14

    def __grain_match__(self, dt1, dt2, grain):
        if grain == "year":
            return dt1.year == dt2.year
        elif grain == "month":
            return dt1.year == dt2.year and dt1.month == dt2.month
        elif grain == "day":
            return dt1.year == dt2.year and dt1.month == dt2.month and dt1.day == dt2.day
        elif grain == "hour":
            return (
                dt1.year == dt2.year and
                dt1.month == dt2.month and
                dt1.day == dt2.day and
                dt1.hour == dt2.hour
            )
        elif grain == "minute":
            return (
                dt1.year == dt2.year and
                dt1.month == dt2.month and
                dt1.day == dt2.day and
                dt1.hour == dt2.hour and
                dt1.minute == dt2.minute
            )
        elif grain == "week":
            return ((dt1-dt2).days<7)
        else:
            raise ValueError(f"Unsupported grain: {grain}")

    def __grain_range_match__(self, dt1, dt2, low_grain, high_grain):
        # Define the hierarchy of grains
        
        if high_grain not in self.grain_order or low_grain not in self.grain_order:
            raise ValueError("Invalid grain level. Supported grains: year, month, day, hour, minute, second")

        high_idx = self.grain_order.index(high_grain)
        low_idx = self.grain_order.index(low_grain)

        if low_idx < high_idx - 1:
            grains_to_compare = self.grain_order[low_idx+1:high_idx+1]
        elif low_idx == high_idx - 1:
            grains_to_compare = [high_grain]
        else:
            return False  # Invalid range: low_grain is not below high_grain
        
        # Compare the relevant grains
        for grain in grains_to_compare:
            if getattr(dt1, grain) != getattr(dt2, grain):
                return False

        return True
    
    def __interval_match__(self, dt_in, dt_start, dt_end):
        return (dt_start<=dt_in and dt_in<dt_end)
    
    def __replace_grain__(self, dt, grain):
        self.grainrep_year
        if grain=='year':
            return dt.replace(year=self.grainrep_year)
        elif grain=='month':
            return dt.replace(year=self.grainrep_year, month=self.grainrep_month)
        elif grain=='day':
            return dt.replace(year=self.grainrep_year, month=self.grainrep_month, day=self.grainrep_day)
        elif grain=='hour':
            return dt.replace(year=self.grainrep_year, month=self.grainrep_month, day=self.grainrep_day, hour=self.grainrep_hour)

    
    def search_point_in_time(self, dt_in, grain):
        indices = []
        count=0
        for i, dt in enumerate(self.datebase):
            if self.__grain_match__(dt_in, dt, grain):
                # if count<10:
                    # print(f"Matched date: {dt}, index: {i}")
                indices.append(i)
                count+=1
        print(f"Total Datetime Matches Found: {count}")
        return indices
    
    def day_of_week_lookup(self, dt_in, high_grain):
        indices = []
        count=0
        for i, dt in enumerate(self.datebase):
            if dt.weekday()==dt_in.weekday() and (self.__grain_range_match__(dt_in, dt, 'day', high_grain) if high_grain in ['hour', 'minute', 'second'] else True):
                # if count<10:
                    # print(WEEKDAY_MAP[dt.weekday()])
                    # print(WEEKDAY_MAP[dow])
                    # print(f"Matched date: {dt}, index: {i}")
                indices.append(i)
                count+=1
        print(f"Total Datetime Matches Found: {count}")
        return indices
    
    def repeat_lookup(self, dt_in, low_grain, high_grain):
        indices = []
        count=0
        for i, dt in enumerate(self.datebase):
            if self.__grain_range_match__(dt_in, dt, low_grain, high_grain):
                # if count<10:
                    # print(f"Matched date: {dt}, index: {i}")
                indices.append(i)
                count+=1
        print(f"Total Datetime Matches Found: {count}")
        return indices
    
    def interval_lookup(self, dt_start=datetime.fromisoformat('1980-01-01T00:00:00+05:30'), dt_end=datetime.fromisoformat("2050-12-31T11:59:59+05:30")):
        indices = []
        count=0
        for i, dt in enumerate(self.datebase):
            if self.__interval_match__(dt, dt_start, dt_end):
                # if count<10:
                    # print(f"Matched date: {dt}, index: {i}")
                indices.append(i)
                count+=1
        print(f"Total Datetime Matches Found: {count}")
        return indices

    def repeat_interval_lookup(self, dt_start, dt_end, rept_freq):
        indices = []
        count=0
        # print(f"Start: {dt_start}, End: {dt_end}")
        
        start_dt = self.__replace_grain__(dt_start, rept_freq)
        end_dt = self.__replace_grain__(dt_end, rept_freq)
        # print(f"Start: {start_dt}, End: {end_dt}")
        for i, dt in enumerate(self.datebase):
            dt2 = self.__replace_grain__(dt, rept_freq)
            # print(f"Date compared: {dt2}")
            if start_dt>end_dt:
                if start_dt<=dt2 or dt2<end_dt:
                    # if count<10:
                    #     print(f"Matched date: {dt}, index: {i}")
                    indices.append(i)
                    count+=1
            else:
                if start_dt<=dt2 and dt2<end_dt:
                    # if count<10:
                    #     print(f"Matched date: {dt}, index: {i}")
                    indices.append(i)
                    count+=1
            
        print(f"Total Datetime Matches Found: {count}")
        return indices
    
    def search(self, responses):
        for response in responses:
            # print(json.dumps(response, indent=4))
            if response['dim']=="time":
                # print(f"Date-Time QUERY: {response['body']}")
                # print(f"Spacy's response: {nlp_with_datetime_ner(query)}")
                # print(f"Contents: {response['value']}")

                # CASE 1 : single value specific year up to any grain level
                if response['value']['type']=='value' and len(response['value']['values'])==1:
                    # print("CASE 1")
                    # print(f"Match the .date() at {response['value']['values'][0]['grain']} grain of {datetime.fromisoformat(response['value']['value'])}")
                    return self.search_point_in_time(dt_in=datetime.fromisoformat(response['value']['value']), grain=response['value']['values'][0]['grain'])
                
                # CASE 2 : multi-value specific year up to any grain level
                if response['value']['type']=='value' and len(response['value']['values'])>1 and datetime.fromisoformat(response['value']['values'][-1]['value']) <= local_now:
                    # print("CASE 2")
                    # print(f"Match the .date() at {response['value']['values'][0]['grain']} grain of {datetime.fromisoformat(response['value']['value'])}")
                    # Calculate delta between two dates
                    return self.search_point_in_time(dt_in=datetime.fromisoformat(response['value']['values'][0]['value']), grain=response['value']['values'][0]['grain'])

                # CASE 3 : multi-value repetitive dates with granularity range
                if response['value']['type']=='value' and len(response['value']['values'])>1 and datetime.fromisoformat(response['value']['values'][-1]['value']) > local_now:
                    # print("CASE 3")
                    # print(f"Match the .date() at {response['value']['values'][0]['grain']} grain of {datetime.fromisoformat(response['value']['value'])}")
                    # Calculate delta between two dates
                    date1 = datetime.fromisoformat(response['value']['values'][0]['value'])
                    date2 = datetime.fromisoformat(response['value']['values'][1]['value'])
                    delta = date2 - date1
                    for rept_freq, num in {'week':7, 'month':30, 'year':365}.items():
                        if delta.days//num==1:
                            # print(f"Repeat frequency: {rept_freq}")
                            if rept_freq=='week':
                                return self.day_of_week_lookup(dt_in=date1, high_grain=response['value']['values'][0]['grain'])
                            if rept_freq=='year' or rept_freq=='month':
                                return self.repeat_lookup(date1, high_grain=response['value']['values'][0]['grain'], low_grain=rept_freq)
                
                # CASE 4 : single value fixed interval
                if response['value']['type']=='interval' and len(response['value']['values'])==1:
                    # print("CASE 4")
                    # print(f"Restrict the .date() at {response['value']['values'][0]['from']['grain'] if 'from' in response['value']['values'][0].keys() else response['value']['values'][0]['to']['grain']} grain between {datetime.fromisoformat(response['value']['values'][0]['from']['value']) if 'from' in response['value']['values'][0].keys() else "???"} and {datetime.fromisoformat(response['value']['values'][0]['to']['value']) if 'to' in response['value']['values'][0].keys() else "???"}")
                    if 'from' in response['value']['values'][0].keys() and 'to' in response['value']['values'][0].keys():
                        return self.interval_lookup(dt_start=datetime.fromisoformat(response['value']['values'][0]['from']['value']), dt_end=datetime.fromisoformat(response['value']['values'][0]['to']['value']))
                    elif 'from' in response['value']['values'][0].keys():
                        return self.interval_lookup(dt_start=datetime.fromisoformat(response['value']['values'][0]['from']['value']))
                    elif 'to' in response['value']['values'][0].keys():
                        return self.interval_lookup(dt_end=datetime.fromisoformat(response['value']['values'][0]['to']['value']))
                
                # CASE 5 : multi-value fixed interval
                if response['value']['type']=='interval' and len(response['value']['values'])>1 and (datetime.fromisoformat(response['value']['values'][-1]['from']['value']) if 'from' in response['value']['values'][-1].keys() else datetime.fromisoformat(response['value']['values'][-1]['to']['value'])) <= local_now:
                    # print("CASE 5")
                    # print(f"Restrict the .date() at {response['value']['values'][0]['from']['grain'] if 'from' in response['value']['values'][0].keys() else response['value']['values'][0]['to']['grain']} grain between {datetime.fromisoformat(response['value']['values'][0]['from']['value']) if 'from' in response['value']['values'][0].keys() else "???"} and {datetime.fromisoformat(response['value']['values'][0]['to']['value']) if 'to' in response['value']['values'][0].keys() else "???"}")
                    return self.interval_lookup(dt_start=datetime.fromisoformat(response['value']['values'][0]['from']['value']), dt_end=datetime.fromisoformat(response['value']['values'][0]['to']['value']))
                
                # CASE 6 : multi-value sliding interval
                if response['value']['type']=='interval' and len(response['value']['values'])>1 and (datetime.fromisoformat(response['value']['values'][-1]['from']['value']) if 'from' in response['value']['values'][-1].keys() else datetime.fromisoformat(response['value']['values'][-1]['to']['value'])) > local_now:
                    # print("CASE 6")
                    # print(f"Restrict the .date() at {response['value']['values'][0]['from']['grain'] if 'from' in response['value']['values'][0].keys() else response['value']['values'][0]['to']['grain']} grain between {datetime.fromisoformat(response['value']['values'][0]['from']['value']) if 'from' in response['value']['values'][0].keys() else "???"} and {datetime.fromisoformat(response['value']['values'][0]['to']['value']) if 'to' in response['value']['values'][0].keys() else "???"}")
                    # print(f"Restrict the .date() at {response['value']['values'][0]['from']['grain']} grain between {datetime.fromisoformat(response['value']['values'][0]['from']['value'])} and {datetime.fromisoformat(response['value']['values'][0]['to']['value'])}")
                    # dt_search.interval_lookup(dt_start=datetime.fromisoformat(response['value']['values'][0]['from']['value']), dt_end=datetime.fromisoformat(response['value']['values'][0]['to']['value']))
                    # Calculate delta between two dates
                    date1from = datetime.fromisoformat(response['value']['values'][0]['from']['value'] if 'from' in response['value']['values'][0].keys() else response['value']['values'][0]['to']['value'])
                    date2 = datetime.fromisoformat(response['value']['values'][1]['from']['value'] if 'from' in response['value']['values'][1].keys() else response['value']['values'][1]['to']['value'])
                    date1to = datetime.fromisoformat(response['value']['values'][0]['to']['value'] if 'to' in response['value']['values'][0].keys() else response['value']['values'][0]['from']['value'])
                    delta = date2 - date1from
                    for rept_freq, num in {'day':1, 'week':7, 'month':30, 'year':365}.items():
                        if delta.days//num==1:
                            # print(f"Repeat frequency: {rept_freq}")
                            if rept_freq in ['year', 'month', 'day']:
                                return self.repeat_interval_lookup(date1from, date1to, rept_freq)
                else:
                    return []

if __name__ == "__main__":

    datetime_data = [datetime.fromisoformat(date) for date in np.load(os.path.join('..', 'datetime_metadata.npy'), allow_pickle=True) if date is not None]
    query = "August vibes"

    duckling = DucklingEngine(port=8010)
    dt_search = DateSearch(datetime_data)
    dtime_indices = dt_search.search(duckling.get_response(query))
    del duckling

#     current_file_dir = os.path.dirname(os.path.abspath(__file__))
#     # embed_store_path = os.path.join(current_file_dir, '..', 'embed_store')
#     # image_dir = os.path.join(current_file_dir, '..', 'ImageSamples')
#     embed_store_path = os.path.join(current_file_dir, '..')
#     image_dir = os.path.join(current_file_dir, '..', '..', '..', 'photos_backup')
#     image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.lower().endswith(('.jpg', 'jpeg', '.png', '.heic', '.heif'))]
    
#     print(f"Total images: {len(image_paths)}")

#     dt_ext = DateTimeExtractor(embedding_dir=embed_store_path)
#     dt_ext.generate_datetime_metadata(image_paths)


