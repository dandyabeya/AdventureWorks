"""Extract the AdventureWorks OLTP tables used by the pipeline into data/raw/*.csv.

Each query flattens one source table (XML demographics via the built-in views,
geography columns into lat/lon). Cleaned data from other sources can replace
these files as long as the column names in SCHEMA.md are kept.
"""
import os
import warnings

import pandas as pd
import pymssql

QUERIES = {
    "sales_order_header": """
        SELECT SalesOrderID, OrderDate, DueDate, ShipDate, Status, OnlineOrderFlag,
               CustomerID, SalesPersonID, TerritoryID, ShipToAddressID, ShipMethodID,
               SubTotal, TaxAmt, Freight, TotalDue
        FROM Sales.SalesOrderHeader""",
    "sales_order_detail": """
        SELECT SalesOrderID, SalesOrderDetailID, ProductID, SpecialOfferID,
               OrderQty, UnitPrice, UnitPriceDiscount, LineTotal
        FROM Sales.SalesOrderDetail""",
    "product": """
        SELECT ProductID, Name, ProductNumber, ProductSubcategoryID, StandardCost, ListPrice,
               Color, Size, ProductLine, Class, Style, MakeFlag, FinishedGoodsFlag,
               SellStartDate, SellEndDate, DiscontinuedDate
        FROM Production.Product""",
    "product_subcategory": """
        SELECT ProductSubcategoryID, ProductCategoryID, Name FROM Production.ProductSubcategory""",
    "product_category": """
        SELECT ProductCategoryID, Name FROM Production.ProductCategory""",
    "product_cost_history": """
        SELECT ProductID, StartDate, EndDate, StandardCost FROM Production.ProductCostHistory""",
    "product_list_price_history": """
        SELECT ProductID, StartDate, EndDate, ListPrice FROM Production.ProductListPriceHistory""",
    "customer": """
        SELECT CustomerID, PersonID, StoreID, TerritoryID FROM Sales.Customer""",
    "store_demographics": """
        SELECT BusinessEntityID AS StoreID, Name AS StoreName, AnnualSales, AnnualRevenue,
               BusinessType, YearOpened, Specialty, SquareFeet, Brands, Internet, NumberEmployees
        FROM Sales.vStoreWithDemographics""",
    "person_demographics": """
        SELECT BusinessEntityID AS PersonID, BirthDate, MaritalStatus, YearlyIncome, Gender,
               TotalChildren, NumberChildrenAtHome, Education, Occupation, HomeOwnerFlag,
               NumberCarsOwned
        FROM Sales.vPersonDemographics""",
    "sales_territory": """
        SELECT TerritoryID, Name, CountryRegionCode, [Group] AS TerritoryGroup
        FROM Sales.SalesTerritory""",
    "address": """
        SELECT AddressID, City, StateProvinceID, PostalCode,
               SpatialLocation.Lat AS Latitude, SpatialLocation.Long AS Longitude
        FROM Person.Address""",
    "state_province": """
        SELECT StateProvinceID, StateProvinceCode, Name, CountryRegionCode, TerritoryID
        FROM Person.StateProvince""",
    "special_offer": """
        SELECT SpecialOfferID, Description, DiscountPct, Type, Category, StartDate, EndDate, MinQty, MaxQty
        FROM Sales.SpecialOffer""",
    "ship_method": """
        SELECT ShipMethodID, Name, ShipBase, ShipRate FROM Purchasing.ShipMethod""",
}


def connect():
    return pymssql.connect(
        server=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", "1433"),
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
    )


def extract_all(raw_dir):
    """Run all queries and write one CSV per table. Returns {table: row_count}."""
    counts = {}
    with connect() as con, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # pandas prefers SQLAlchemy connections
        for name, sql in QUERIES.items():
            df = pd.read_sql(sql, con)
            df.to_csv(raw_dir / f"{name}.csv", index=False)
            counts[name] = len(df)
    return counts
