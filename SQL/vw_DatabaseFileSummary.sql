USE [WyethDBA]
GO
IF  EXISTS (SELECT * FROM sys.views WHERE object_id = OBJECT_ID(N'[dbo].[vw_DatabaseFileSummary]'))
DROP VIEW [dbo].[vw_DatabaseFileSummary]
GO
SET ANSI_NULLS ON
GO
SET QUOTED_IDENTIFIER ON
GO
Create  View [dbo].[vw_DatabaseFileSummary]
As
/*********************************************************************/
/* Database: WyethDBA  											     */
/* Purpose: Display All Active Database/Log Space Allocation 	     */
/*                                                                   */
/*  Org. Date: 01/13/2009        EPAK                                */
/*  Mod. Date: 00/00/0000                                            */
/*********************************************************************/
Select	DIF.JobDateInfo As LastProcessedDate,
		DFS.SiteID,
		DFS.ServerID,
		DSV.ServerVersionDesc,
		DST.ServerTypeDesc,
		SIT.SiteDesc As [SiteDescription],
		DSR.ServerName As [ServerName],
		NAM.DatabaseName,
		DFS.FileType,
		DFS.Data_Log_Size_MB,
		DFS.GrowthLimit_MB,
		DFS.PercentFree

From	DatabaseFileSummary DFS INNER JOIN DateInfo DIF
ON		DFS.DateID = DIF.DateID
		INNER JOIN DatabaseSite SIT
ON		DFS.SiteID = SIT.SiteID
		INNER JOIN DatabaseServer DSR
ON		DFS.ServerID = DSR.ServerID
		INNER JOIN DatabaseServerVersion DSV
ON		DSR.ServerVersionID = DSV.ServerVersionID
		INNER JOIN DatabaseServerType DST
ON		DSR.ServerTypeID = DST.ServerTypeID
		INNER JOIN DatabaseName NAM
ON		DFS.DatabaseNameID = NAM.DatabaseNameID
Where	DFS.DateID = (Select MAX(DateID) From DateInfo)
And		DSR.ActiveFlag = 'Y'
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
GO
GRANT  SELECT  ON [dbo].[vw_DatabaseFileSummary]  TO [WyethDBA_Client]