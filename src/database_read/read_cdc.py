import pandas as pd

class CountyData:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)
        self.clean_data()

    def get_summary(self):
        """
        Returns a summary of the data for the county.
        """
        summary = self.data_frame.describe()
        return summary
    
    def clean_data(self):
        """
        Cleans the DataFrame by handling missing values.
        Fixes capitalization issues.
        """
        self.data.replace('#', 0, inplace=True)
        self.data.fillna(0, inplace=True)
        self.data.columns = [col.title() for col in self.data.columns]

        county_col = self.data.columns[0]
        self.data[county_col] = self.data[county_col].astype(str).str.title()

    def get_county_names(self):
        """
        Returns a list of all county names in the DataFrame and exports it as a .csv file.
        """
        county_col = self.data.columns[0]
        county_names = self.data[county_col].astype(str).tolist()
        exclude = {'total'}
        county_names = [name for name in county_names if str(name).strip().lower() not in exclude]

        pd.DataFrame(county_names, columns=[county_col]).to_csv("county_names.csv", index=False, header=False)
        return county_names

    def get_specific_value(self, county_name, column_name):
        """
        Return a specific value from the DataFrame based on county name and column name.
        """
        row = self.data.loc[self.data.iloc[:, 0] == county_name]
        if row.empty:
            raise ValueError(f"County '{county_name}' not found.")
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found.")
        return row[column_name].values[0]
    
    def get_column(self, column_name):
        """
        Returns a specific column from the DataFrame.
        """
        if column_name not in self.data.columns:
            raise ValueError(f"Column '{column_name}' not found.")
        return self.data[column_name]
    
    def get_county_row(self, county_name):
        """
        Returns the entire row (all features) for a specific county.
        """
        row = self.data.loc[self.data.iloc[:, 0] == county_name]
        if row.empty:
            raise ValueError(f"County '{county_name}' not found.")
        return row.squeeze()
    
    
if __name__ == "__main__":
    # Example usage
    csv_file = "/Users/andrewjin/Documents/GitHub/alternative-data-NC-hospitals/data/NC Medicaid Reports/SFY2023_Annual_Unduplicated_Enrollment_Counts_by_County_and_Budget_Group (1).xlsx"
    county_data = CountyData(csv_file)
    county_list = county_data.get_county_names()

    # Access examples
    print(county_data.get_specific_value("Burke", "County Total"))
    print(county_data.get_column("County Total").head())
    print(county_data.get_county_row("Durham"))






